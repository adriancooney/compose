from __future__ import absolute_import
from __future__ import unicode_literals

import re
import sys
import time
import curses
from threading import Timer
from math import floor
from collections import namedtuple
from itertools import cycle
from threading import Thread

from docker.errors import APIError
from six.moves import _thread as thread
from six.moves.queue import Empty
from six.moves.queue import Queue

from . import colors
from compose import utils
from compose.cli.signals import ShutdownException
from compose.utils import split_buffer


class LogPresenter(object):

    def __init__(self, prefix_width, color_func):
        self.prefix_width = prefix_width
        self.color_func = color_func

    def present(self, container, line):
        prefix = container.name_without_project.ljust(self.prefix_width)
        return '{prefix} {line}'.format(
            prefix=self.color_func(prefix + ' |'),
            line=line)


def build_log_presenters(service_names, monochrome):
    """Return an iterable of functions.

    Each function can be used to format the logs output of a container.
    """
    prefix_width = max_name_width(service_names)

    def no_color(text):
        return text

    for color_func in cycle([no_color] if monochrome else colors.rainbow()):
        yield LogPresenter(prefix_width, color_func)


def max_name_width(service_names, max_index_width=3):
    """Calculate the maximum width of container names so we can make the log
    prefixes line up like so:

    db_1  | Listening
    web_1 | Listening
    """
    return max(len(name) for name in service_names) + max_index_width


class LogPrinter(object):
    """Print logs from many containers to a single output stream."""

    def __init__(self,
                 containers,
                 presenters,
                 event_stream,
                 output=sys.stdout,
                 cascade_stop=False,
                 log_args=None):
        self.containers = containers
        self.presenters = presenters
        self.event_stream = event_stream
        self.output = utils.get_output_stream(output)
        self.cascade_stop = cascade_stop
        self.log_args = log_args or {}

    def run(self):
        if not self.containers:
            return

        queue = Queue()
        thread_args = queue, self.log_args
        thread_map = build_thread_map(self.containers, self.presenters, thread_args)
        start_producer_thread((
            thread_map,
            self.event_stream,
            self.presenters,
            thread_args))

        for line in consume_queue(queue, self.cascade_stop):
            remove_stopped_threads(thread_map)

            if not line:
                if not thread_map:
                    # There are no running containers left to tail, so exit
                    return
                # We got an empty line because of a timeout, but there are still
                # active containers to tail, so continue
                continue

            self.output.write(line)
            self.output.flush()

class GridPrinter(object):
    def __init__(self,
                 containers,
                 presenters,
                 event_stream,
                 output=sys.stdout,
                 cascade_stop=False,
                 log_args=None):
        self.containers = containers
        self.presenters = presenters
        self.event_stream = event_stream
        self.cascade_stop = cascade_stop
        self.log_args = log_args or {}
        self.deferred_render = None

    def grid_log(self, root_window):
        self.root_window = root_window
        self.container_windows = []
        self.logs = {}

        # Turn off the cursor
        curses.curs_set(0)
        curses.use_default_colors()
        for i in range(0, curses.COLORS):
            curses.init_pair(i, i, -1);

        self.wh, self.ww = root_window.getmaxyx()
        for i, container in enumerate(self.containers):
            x, y, width, height = self.get_window_size(root_window, i)
            self.container_windows.append(curses.newwin(height, width, y, x))
            self.logs[container.name_without_project] = []

        self.render(root_window)

        queue = Queue()
        thread_args = queue, self.log_args
        thread_map = build_thread_map(self.containers, self.presenters, thread_args)
        start_producer_thread((thread_map, self.event_stream, self.presenters, thread_args))

        for line in consume_queue(queue, self.cascade_stop):
            remove_stopped_threads(thread_map)

            if not line:
                if not thread_map:
                    # There are no running containers left to tail, so exit
                    return
                # We got an empty line because of a timeout, but there are still
                # active containers to tail, so continue
                continue

            self.handle_line(line)

    def render(self, root_window):
        root_window.erase()

        wh, ww = root_window.getmaxyx()
        resized = wh != self.wh or ww != self.ww

        if resized:
            self.wh, self.ww = wh, ww

        for i, log in enumerate(zip(self.containers, self.container_windows)):
            container, window = log

            if resized:
                x, y, width, height = self.get_window_size(root_window, i)
                window.mvwin(y, x)
                window.resize(height, width)

            window.attrset(curses.color_pair((i % curses.COLORS) + 1))
            window.border()
            window.addstr(0, 3, " %s " % container.name_without_project)
            window.attrset(0)
            window.overlay(root_window)

            wh, ww = window.getmaxyx()
            logs = [log for log in self.get_logs(container.name_without_project, wh - 2, ww - 4)]

            for i, log in enumerate(logs):
                window.addstr(len(logs) - i, 2, log.ljust(ww - 2))

        root_window.refresh()


    def get_logs(self, container, lines, width):
        log = self.logs[container]
        log_count = 1
        line_count = 0

        if len(log) == 0:
            raise StopIteration

        # While we can still send lines and there is logs left
        while line_count < lines and len(log) > log_count:
            line = log[-log_count]
            log_count += 1

            for wrapped in reversed(textwrap(line.encode("ascii", "ignore"), width)):
                if line_count < lines:
                    line_count += 1
                    yield wrapped

    def get_window_size(self, root_window, window_position):
        """Return a tuple of (pos_x, pos_w, size_x, size_y)"""
        window_count = len(self.containers)
        mh, mw = root_window.getmaxyx()
        hw, hh = mw/2, mh/2

        panel_full_size = (0, 0, mw, mh)
        panel_left_half = (0, 0, hw, mh)
        panel_right_half = (hw, 0, hw, mh)
        panel_top_left = (0, 0, hw, hh)
        panel_bottom_left = (0, hh, hw, hh)
        panel_top_right = (hw, 0, hw, hh)
        panel_bottom_right = (hw, hh, hw, hh)

        layouts = [
            [panel_full_size],
            [panel_left_half, panel_right_half],
            [panel_left_half, panel_top_right, panel_bottom_right],
            [panel_top_left, panel_bottom_left, panel_top_right, panel_bottom_right]
        ]

        return layouts[window_count - 1][window_position]

    def handle_line(self, line):
        container_name, sep, log = line.partition("|")
        container_name = strip_ansi(container_name.strip())
        log = log[5:] # Remove the leading "| [0m"

        self.logs[container_name].append(log)

        # Cap the render to 30 fps
        def do_render():
            self.render(self.root_window)
            self.deferred_render = None

        if self.deferred_render:
            self.deferred_render.cancel()

        self.deferred_render = Timer(1.0 / 30, do_render)
        self.deferred_render.start()

    def run(self):
        if not self.containers:
            return

        curses.wrapper(self.grid_log)

def strip_ansi(str):
    ansi_escape = re.compile(r'\x1b[^m]*m')
    return ansi_escape.sub('', str)

def textwrap(str, width):
    length = len(str)
    lines = (length / width) + 1
    split = []

    for i in range(lines):
        pos = i * width
        lim = min(length - 1, pos + width)
        line = str[pos:lim]

        for spl in line.split("\n"):
            split.append(spl)

    return split


def remove_stopped_threads(thread_map):
    for container_id, tailer_thread in list(thread_map.items()):
        if not tailer_thread.is_alive():
            thread_map.pop(container_id, None)


def build_thread(container, presenter, queue, log_args):
    tailer = Thread(
        target=tail_container_logs,
        args=(container, presenter, queue, log_args))
    tailer.daemon = True
    tailer.start()
    return tailer


def build_thread_map(initial_containers, presenters, thread_args):
    return {
        container.id: build_thread(container, next(presenters), *thread_args)
        for container in initial_containers
    }


class QueueItem(namedtuple('_QueueItem', 'item is_stop exc')):

    @classmethod
    def new(cls, item):
        return cls(item, None, None)

    @classmethod
    def exception(cls, exc):
        return cls(None, None, exc)

    @classmethod
    def stop(cls):
        return cls(None, True, None)


def tail_container_logs(container, presenter, queue, log_args):
    generator = get_log_generator(container)

    try:
        for item in generator(container, log_args):
            queue.put(QueueItem.new(presenter.present(container, item)))
    except Exception as e:
        queue.put(QueueItem.exception(e))
        return

    if log_args.get('follow'):
        queue.put(QueueItem.new(presenter.color_func(wait_on_exit(container))))
    queue.put(QueueItem.stop())


def get_log_generator(container):
    if container.has_api_logs:
        return build_log_generator
    return build_no_log_generator


def build_no_log_generator(container, log_args):
    """Return a generator that prints a warning about logs and waits for
    container to exit.
    """
    yield "WARNING: no logs are available with the '{}' log driver\n".format(
        container.log_driver)


def build_log_generator(container, log_args):
    # if the container doesn't have a log_stream we need to attach to container
    # before log printer starts running
    if container.log_stream is None:
        stream = container.logs(stdout=True, stderr=True, stream=True, **log_args)
    else:
        stream = container.log_stream

    return split_buffer(stream)


def wait_on_exit(container):
    try:
        exit_code = container.wait()
        return "%s exited with code %s\n" % (container.name, exit_code)
    except APIError as e:
        return "Unexpected API error for %s (HTTP code %s)\nResponse body:\n%s\n" % (
            container.name, e.response.status_code,
            e.response.text or '[empty]'
        )


def start_producer_thread(thread_args):
    producer = Thread(target=watch_events, args=thread_args)
    producer.daemon = True
    producer.start()


def watch_events(thread_map, event_stream, presenters, thread_args):
    for event in event_stream:
        if event['action'] == 'stop':
            thread_map.pop(event['id'], None)

        if event['action'] != 'start':
            continue

        if event['id'] in thread_map:
            if thread_map[event['id']].is_alive():
                continue
            # Container was stopped and started, we need a new thread
            thread_map.pop(event['id'], None)

        thread_map[event['id']] = build_thread(
            event['container'],
            next(presenters),
            *thread_args)


def consume_queue(queue, cascade_stop):
    """Consume the queue by reading lines off of it and yielding them."""
    while True:
        try:
            item = queue.get(timeout=0.1)
        except Empty:
            yield None
            continue
        # See https://github.com/docker/compose/issues/189
        except thread.error:
            raise ShutdownException()

        if item.exc:
            raise item.exc

        if item.is_stop:
            if cascade_stop:
                raise StopIteration
            else:
                continue

        yield item.item
