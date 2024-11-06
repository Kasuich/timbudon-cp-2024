import time
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from queue import Empty, Full
from threading import Thread
from typing import Optional

import zmq
from loguru import logger


class NeedRestartException(Exception):
    pass


@dataclass
class ZmqConfigPub:
    ADDRESS: str
    RESTART_SLEEP: int = 1000
    FULL_QUEUE_SLEEP: int = 100
    NO_MSG_SLEEP: int = 100


@dataclass
class ZmqConfigSub:
    ADDRESS: str
    RESTART_SLEEP: int = 1000
    FULL_QUEUE_SLEEP: int = 100
    NO_MSG_SLEEP: int = 100


class ZmqSub(Thread):

    def __init__(
        self,
        config: ZmqConfigSub,
        service_name: str,
        outp_queue: Queue,
        stop_event: Event,
        bind: bool,
    ):

        self._cfg = config
        self.service_name = service_name
        self._current_msg = None
        self._outp_queue = outp_queue
        self._stop_event = stop_event
        self.__restart = True
        self.__bind = bind
        self.__context = zmq.Context()
        self.__socket = None

        super().__init__()

    def _start_action(self) -> bool:

        logger.info(f"{self.service_name} is starting")
        try:
            if self.__socket is not None:
                self.__socket.close()
        except Exception as e:
            logger.exception(f"{self.service_name}. {e}")
        try:
            self.__socket = self.__context.socket(zmq.SUB)
            self.__socket.setsockopt(zmq.SUBSCRIBE, b"")
            if self.__bind:
                self.__socket.bind(f"{self._cfg.ADDRESS}")
            else:
                self.__socket.connect(f"{self._cfg.ADDRESS}")
            logger.info(f"{self.service_name} is started")
            return True
        except Exception as e:
            logger.exception(f"{self.service_name}. {e}")
            return False

    def _stop_action(self) -> bool:
        try:
            if self.__socket is not None:
                logger.info(f"{self.service_name} is stopping")
                self.__socket.close()
                self.__socket = None
                logger.info(f"{self.service_name} is stopped")
        except Exception as e:
            logger.exception(f"{self.service_name}. {e}")

    def _get_message(self) -> Optional[any]:
        try:
            msg = self.__socket.recv_json(flags=zmq.NOBLOCK)
            logger.debug(
                f"""{self.service_name}:{self.service_name} receive message {msg} """
            )
            return msg
        except zmq.ZMQError as e:
            if e.errno == zmq.Errno.EAGAIN:
                return None
            else:
                logger.exception(f"{self.service_name}. {e}")
                raise NeedRestartException

    def run(self):
        while not self._stop_event.is_set():
            self.__update_stat()
            if self.__restart:
                try:
                    self._stop_action()
                    if self._start_action():
                        self.__restart = False
                    else:
                        raise Exception
                except Exception as e:
                    logger.exception(f"{self.service_name}. {e}")
                    time.sleep(self._cfg.RESTART_SLEEP)
            else:
                try:
                    msg = (
                        self._get_message()
                        if self._current_msg is None
                        else self._current_msg
                    )
                except NeedRestartException as e:
                    self.__restart = True
                    logger.exception(f"{self.service_name}. {e}")
                    continue
                if msg is not None:
                    self.__message_counter = self.__message_counter + 1
                    try:
                        self._outp_queue.put_nowait(msg)
                        self._current_msg = None
                    except Full as e:
                        self._current_msg = msg
                        logger.exception(f"{self.service_name}. {e}")
                        time.sleep(self._cfg.FULL_QUEUE_SLEEP)
                else:
                    time.sleep(self._cfg.NO_MSG_SLEEP)

        self._stop_action()


class ZmqPub(Thread):

    def __init__(
        self,
        config: ZmqConfigPub,
        service_name: str,
        inp_queue: Queue,
        stop_event: Event,
    ):

        self._cfg = config
        self.service_name = service_name
        self._current_msg = None
        self._inp_queue = inp_queue
        self._stop_event = stop_event
        self.__restart = True

        super().__init__()

    def _start_action(self) -> bool:

        logger.info(f"{self.service_name} is starting")
        try:
            if self.__socket is not None:
                self.__socket.close()
        except Exception as e:
            logger.exception(f"{self.service_name}. {e}")
        try:
            self.__socket = self.__context.socket(zmq.PUB)
            if self.__bind:
                self.__socket.bind(f"{self._cfg.ADDRESS}")
            else:
                self.__socket.connect(f"{self._cfg.ADDRESS}")
            logger.info(f"{self.service_name} is started")
            return True
        except Exception as e:
            logger.exception(f"{self.service_name}. {e}")
            return False

    def _stop_action(self) -> bool:
        try:
            if self.__socket is not None:
                logger.info(f"{self.service_name} is stopping")
                self.__socket.close()
                self.__socket = None
                logger.info(f"{self.service_name} is stopped")
        except Exception as e:
            logger.exception(f"{self.service_name}. {e}")

    def _put_message(self, msg) -> bool:

        try:
            self.__socket.send_json(msg, flags=0)
            logger.debug(f"""{self.service_name}. sent message {msg}""")
            return True
        except Exception as e:
            logger.exception(f"{msg}. {e}", exc_info=True)
            logger.exception(f"{self.service_name}. {e}", exc_info=True)
            return False

    def run(self):
        while not self._stop_event.is_set():
            self.__update_stat()
            if self.__restart:
                try:
                    self._stop_action()
                    if self._start_action():
                        self.__restart = False
                    else:
                        raise Exception
                except Exception as e:
                    logger.exception(f"{self.service_name}. {e}")
                    time.sleep(self._cfg.RESTART_SLEEP)
            else:
                try:
                    msg = (
                        self._inp_queue.get_nowait()
                        if self._current_msg is None
                        else self._current_msg
                    )
                except Empty:
                    time.sleep(self._cfg["NO_MSG_SLEEP"])
                    continue
                if not self._put_message(msg):
                    self._current_msg = msg
                    logger.error(f"{self.service_name} sending error. Need restart")
                    self.__restart = True
                    continue
                else:
                    self._current_msg = None
            self._stop_action()

        self._stop_action()
