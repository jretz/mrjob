# -*- coding: utf-8 -*-
# Copyright 2011 Matthew Tai and Yelp
# Copyright 2012-2013 Yelp and Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an MRJob in RAM by running all mappers and reducers through the same
process. Useful for debugging."""
from __future__ import with_statement

__author__ = 'Matthew Tai <mtai@adku.com>'

import collections
import logging
import os

try:
    from cStringIO import StringIO
    StringIO  # quiet "redefinition of unused ..." warning from pyflakes
except ImportError:
    from StringIO import StringIO

from mrjob.job import MRJob
from mrjob.parse import parse_mr_job_stderr
from mrjob.protocol import InRAMProtocol
from mrjob.sim import SimMRJobRunner
from mrjob.sim import SimRunnerOptionStore
from mrjob.util import save_current_environment
from mrjob.util import save_cwd

log = logging.getLogger(__name__)


# Deprecated in favor of class variables, remove in v0.5.0
DEFAULT_MAP_TASKS = 1
DEFAULT_REDUCE_TASKS = 1


class InRAMMRJobRunner(SimMRJobRunner):
    """Runs an :py:class:`~mrjob.job.MRJob` in the same process, so it's easy
    to attach a debugger.

    This is the default way to run jobs (we assume you'll spend some time
    debugging your job before you're ready to run it on EMR or Hadoop).

    To more accurately simulate your environment prior to running on
    Hadoop/EMR, use ``-r local`` (see
    :py:class:`~mrjob.local.LocalMRJobRunner`).
    """
    alias = 'inram'

    OPTION_STORE_CLASS = SimRunnerOptionStore

    # stick to a single split for efficiency
    _DEFAULT_MAP_TASKS = 1
    _DEFAULT_REDUCE_TASKS = 1

    def __init__(self, mrjob_cls=None, **kwargs):
        """:py:class:`~mrjob.inram.InRAMMRJobRunner` takes the same
        keyword args as :py:class:`~mrjob.runner.MRJobRunner`. However, please
        note:

        * *hadoop_extra_args*, *hadoop_input_format*, *hadoop_output_format*,
          and *hadoop_streaming_jar*, and *partitioner* are ignored
          because they require Java. If you need to test these, consider
          starting up a standalone Hadoop instance and running your job with
          ``-r hadoop``.
        * *python_bin*, *setup*, *setup_cmds*, *setup_scripts* and
          *steps_python_bin* are ignored because we don't invoke
          subprocesses.
        """
        super(InRAMMRJobRunner, self).__init__(**kwargs)
        assert ((mrjob_cls) is None or issubclass(mrjob_cls, MRJob))

        self._mrjob_cls = mrjob_cls

        # Force the internal protocol to InRAMProtocol. We don't want any
        # encoding/decoding going on here.
        mrjob_cls.INTERNAL_PROTOCOL = InRAMProtocol

    # options that we ignore because they involve running subprocesses
    IGNORED_LOCAL_OPTS = [
        'bootstrap_mrjob',
        'python_bin',
        'setup',
        'setup_cmds',
        'setup_scripts',
        'steps_python_bin',
    ]

    def _check_step_works_with_runner(self, step_dict):
        for key in ('mapper', 'combiner', 'reducer'):
            if key in step_dict:
                substep = step_dict[key]
                if substep['type'] != 'script':
                    raise Exception(
                        "InRAMMRJobRunner cannot run %s steps." %
                        substep['type'])
                if 'pre_filter' in substep:
                    raise Exception(
                        "InRAMMRJobRunner cannot run filters.")

    def _run_steps(self):
        self.read_store = InMemoryStore()
        self.write_store = InMemoryStore()

        self.read_from_ram = False
        self.write_to_ram = True

        # run mapper, combiner, sort, reducer for each step
        for step_num, step in enumerate(self._get_steps()):
            self._check_step_works_with_runner(step)
            self._counters.append({})

            if step_num + 1 == self._num_steps():
                if 'reducer' not in step:
                    self.write_to_ram = False

            self._invoke_step(step_num, 'mapper')

            if step_num == 0:
                self.read_from_ram = True

            if step_num + 1 == self._num_steps():
                self.write_to_ram = False

            if 'reducer' in step:
                # sort the output. Treat this as a mini-step for the purpose
                # of self._prev_outfiles
                sort_output_path = os.path.join(
                    self._get_local_tmp_dir(),
                    'step-%d-mapper-sorted' % step_num)

                self._invoke_sort(self._step_input_paths(), sort_output_path)
                self._prev_outfiles = [sort_output_path]

                # run the reducer
                self._invoke_step(step_num, 'reducer')

    def per_step_runner_finish(self, step_num):
        """ Runner specific method to be executed to mark the step completion.
        """
        self.read_store = self.write_store
        self.write_store = InMemoryStore()

    def _create_setup_wrapper_script(self):
        # In RAM mode does not use a wrapper script (no subprocesses)
        pass

    def warn_ignored_opts(self):
        """ Warn the user of opts being ignored by this runner.
        """
        super(InRAMMRJobRunner, self).warn_ignored_opts()
        for ignored_opt in self.IGNORED_LOCAL_OPTS:
            if ((not self._opts.is_default(ignored_opt)) and
                    self._opts[ignored_opt]):
                log.warning('ignoring %s option (use -r local instead): %r' %
                            (ignored_opt, self._opts[ignored_opt]))

    def _get_steps(self):
        """Redefine this so that we can get step descriptions without
        calling a subprocess."""
        if self._steps is None:
            job_args = ['--steps'] + self._mr_job_extra_args(local=True)
            self._steps = self._mrjob_cls(args=job_args)._steps_desc()

        return self._steps

    def _run_step(self, step_num, step_type, input_path, output_path,
                  working_dir, env, child_stdin=None):
        step = self._get_step(step_num)

        common_args = (['--step-num=%d' % step_num] +
                       self._mr_job_extra_args(local=True))

        if step_type == 'mapper':
            child_args = (
                ['--mapper'] + [input_path] + common_args)
        elif step_type == 'reducer':
            child_args = (
                ['--reducer'] + [input_path] + common_args)
        elif step_type == 'combiner':
            child_args = ['--combiner'] + common_args + ['-']

        child_instance = self._mrjob_cls(args=child_args)

        has_combiner = (step_type == 'mapper' and 'combiner' in step)

        # Use custom stdin
        if has_combiner:
            child_stdout = StringIO()
        else:
            child_stdout = open(output_path, 'w')

        if self.read_from_ram:
            self._mrjob_cls._DEFAULT_READER = self.read_store.read
        else:
            self._mrjob_cls._DEFAULT_READER = None

        if self.write_to_ram:
            self._mrjob_cls._DEFAULT_WRITER = self.write_store.write
        else:
            self._mrjob_cls._DEFAULT_WRITER = None

        with save_current_environment():
            with save_cwd():
                os.environ.update(env)
                os.chdir(working_dir)

                child_instance.sandbox(stdin=child_stdin, stdout=child_stdout)
                child_instance.execute()

        if has_combiner:
            sorted_lines = sorted(child_stdout.getvalue().splitlines())
            combiner_stdin = StringIO('\n'.join(sorted_lines))
        else:
            child_stdout.flush()

        child_stdout.close()

        while len(self._counters) <= step_num:
            self._counters.append({})
        parse_mr_job_stderr(child_instance.stderr.getvalue(),
                            counters=self._counters[step_num])

        if has_combiner:
            self._run_step(step_num, 'combiner', None, output_path,
                           working_dir, env, child_stdin=combiner_stdin)

            combiner_stdin.close()


class InMemoryStore(object):
    def __init__(self):
        self.store = collections.defaultdict(list)

    def read(self):
        for key, values in self.store.iteritems():
            for value in values:
                yield key, value

    def write(self, output):
        key, value = output
        self.store[key].append(value)
