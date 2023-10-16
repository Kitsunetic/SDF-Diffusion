from datetime import datetime
from functools import reduce
from pathlib import Path


class CustomLogger:
    def __init__(self, filename=None, filemode="a", use_color=True, lock=False):
        self.lock = lock
        self.empty = True

        if not lock:
            if filename is not None:
                self.empty = False
                filename = Path(filename)
                if filename.is_dir():
                    timestr = self._get_timestr().replace(" ", "_").replace(":", "-")
                    filename = filename / f"log_{timestr}.log"
                self.file = open(filename, filemode)
            else:
                self.empty = True

            self.use_color = use_color

    def _get_timestr(self):
        n = datetime.now()
        return f"{n.year - 2000:02d}:{n.month:02d}:{n.day:02d} {n.hour:02d}:{n.minute:02d}:{n.second:02d}"

    def _write(self, msg, level):
        if self.lock:
            return

        timestr = self._get_timestr()
        out = f"[{timestr} {level}] {msg}"

        if self.use_color:
            if level == " INFO":
                # print("\033[32m" + out + "\033[0m")
                # print("\033[33m" + out + "\033[0m")
                # print("\033[34m" + out + "\033[0m")
                print("\033[96m" + out + "\033[0m")
                # print("\033[91m" + out + "\033[0m")
            elif level == " WARN":
                print("\033[35m" + out + "\033[0m")
            elif level == "ERROR":
                print("\033[31m" + out + "\033[0m")
            elif level == "FATAL":
                print("\033[43m\033[1m" + out + "\033[0m")
            else:
                print(out)
        else:
            print(out)

        if not self.empty:
            self.file.write(out + "\r\n")

    def debug(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "DEBUG")

    def info(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " INFO")

    def warn(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " WARN")

    def error(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "ERROR")

    def fatal(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "FATAL")

    def flush(self):
        if not self.lock and not self.empty:
            self.file.flush()


def timenow(braket=False):
    n = datetime.now()
    if braket:
        return f"[{n.year}-{n.month:02d}-{n.day:02d} {n.hour:02d}:{n.minute:02d}:{n.second:02d}]"
    else:
        return f"{n.year}-{n.month:02d}-{n.day:02d} {n.hour:02d}:{n.minute:02d}:{n.second:02d}"


_logger = CustomLogger()


def basicConfig(filename, lock=False):
    _logger.__init__(filename, lock=lock)


def getLogger():
    return _logger
