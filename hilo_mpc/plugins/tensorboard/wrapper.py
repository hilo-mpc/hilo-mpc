#   
#   This file is part of HILO-MPC
#
#   HILO-MPC is a toolbox for easy, flexible and fast development of machine-learning-supported
#   optimal control and estimation problems
#
#   Copyright (c) 2021 Johannes Pohlodek, Bruno Morabito, Rolf Findeisen
#                      All rights reserved
#
#   HILO-MPC is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   HILO-MPC is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with HILO-MPC. If not, see <http://www.gnu.org/licenses/>.
#

from multiprocessing import Process
import os
import sys


class _TensorBoardSupervisor:
    """https://stackoverflow.com/a/60021949"""
    def __init__(self, log_dir, browser):
        """Constructor method"""
        self._server = _TensorBoardServer(log_dir)
        self._server.start()
        print("Started TensorBoard server")
        if browser == 'chrome':
            self._browser = _ChromeProcess()
        elif browser == 'firefox':
            self._browser = _FirefoxProcess()
        else:
            raise NotImplementedError(f"Support for browser '{browser}' not yet implemented")
        print("Started selected browser")
        self._browser.start()

    def finalize(self):
        """

        :return:
        """
        if self._server.is_alive():
            print("Killing TensorBoard server")
            self._browser.terminate()
            self._server.terminate()
            self._server.join()


class _TensorBoardServer(Process):
    """https://stackoverflow.com/a/60021949"""
    def __init__(self, log_dir):
        """Constructor method"""
        super().__init__()
        self._os_name = os.name
        self._log_dir = str(log_dir)

    def run(self) -> None:
        """

        :return:
        """
        if self._os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self._log_dir}" 2> NUL')
        elif self._os_name == 'posix':  # Linux
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self._log_dir}" --host `hostname -I` >/dev/null'
                      f' 2>&1')
        else:
            raise NotImplementedError(f"Support for OS '{self._os_name}' not yet implemented")


class _ChromeProcess(Process):
    """https://stackoverflow.com/a/60021949"""
    def __init__(self):
        """Constructor method"""
        super().__init__()
        self._os_name = os.name
        self.daemon = True

    def run(self) -> None:
        """

        :return:
        """
        if self._os_name == 'nt':  # Windows
            os.system(f'start chrome http://localhost:6006/')
        elif self._os_name == 'posix':  # Linux
            os.system('google-chrome http://$(hostname):6006/')
        else:
            raise NotImplementedError(f"Support for OS '{self._os_name}' not yet implemented")


class _FirefoxProcess(Process):
    """"""
    def __init__(self):
        """Constructor method"""
        super().__init__()
        self._os_name = os.name
        self.daemon = True

    def run(self) -> None:
        """

        :return:
        """
        if self._os_name == 'nt':  # Windows
            os.system('start firefox http://localhost:6006/')
        elif self._os_name == 'posix':  # Linux
            os.system('firefox http://localhost:6006/')
        else:
            raise NotImplementedError(f"Support for OS '{self._os_name}' not yet implemented")


class _TensorBoardWrapper:
    """"""
    def __init__(self, *args, **kwargs):
        """Constructor method"""
        self._log_dir = kwargs.get('log_dir')
        self._supervisor = None

    def _process_comment(self, prefix, **kwargs):
        """

        :param kwargs:
        :return:
        """
        comment = prefix + '-' + '-'.join(f'{key}={val}' for key, val in kwargs.items())
        path = os.path.join(self._log_dir, comment)  # TODO: What would happen here, if self._log_dir was None
        if not os.path.exists(path):
            os.makedirs(path)
        run_no = len(next(os.walk(path))[1])
        path += f'/run_{run_no + 1}'

        return comment, path

    @property
    def summary_writer(self):
        """

        :return:
        """
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter

    @property
    def callback(self):
        """

        :return:
        """
        from tensorflow.keras.callbacks import TensorBoard

        return TensorBoard

    def get_summary_writer(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        comment, path = self._process_comment('pt', **kwargs)
        if self._log_dir is not None:
            return self.summary_writer(log_dir=path)
        else:
            return self.summary_writer(comment=comment)

    def get_callback(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        _, path = self._process_comment('tf', **kwargs)
        if self._log_dir is not None:
            return self.callback(log_dir=path)

    def show(self, browser):
        """

        :param browser:
        :return:
        """
        self._supervisor = _TensorBoardSupervisor(self._log_dir, browser)

    def close(self):
        """

        :return:
        """
        if self._supervisor is not None:
            self._supervisor.finalize()


def get_wrapper(*args, **kwargs):
    """

    :param args:
    :param kwargs:
    :return:
    """
    return _TensorBoardWrapper(*args, **kwargs)


__all__ = [
    'get_wrapper'
]
