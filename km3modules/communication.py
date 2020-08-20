# Filename: communication.py
# -*- coding: utf-8 -*-
# pylint: disable=locally-disabled
"""
Talk to the ELOG, interact with the Chat server etc.

"""

import builtins
import os
import passlib
import requests
import time

import km3pipe as kp


class ELOGService(kp.Module):
    """Allows communication with the KM3NeT ELOG server.

    Parameters
    ----------
    username: string (optional)
    password: string (required)
    url: string (optional)

    """

    def configure(self):
        self.url = self.get("url", default="https://elog.km3net.de/")
        self._password = create_elog_password_hash(self.require("password"))
        self._username = self.get("username", default="km3net")
        self.expose(self.post_elog, "post_elog")

    def post_elog(
        self, logbook, subject, message, author, message_type="Comment", files=None
    ):
        """Post an ELOG entry

        Make sure to provide an existing `author`.

        Parameters
        ----------
        logbook: string (e.g. "individual+Logbooks")
        subject: string
        message: string
        author: string (make sure to provide an existing one!)
        message_type: string (default="Comment")
        files: list of strings (filenames, optional)
        """
        data = {
            "exp": logbook,
            "cmd": "Submit",
            "Subject": subject,
            "Author": author,
            "Type": message_type,
            "unm": self._username,
            "upwd": self._password,
            "When": int(time.time()),
        }
        if files is None:
            files = []
        files = [
            ("attfile{}".format(idx + 1), (os.path.basename(f), builtins.open(f, "rb")))
            for idx, f in enumerate(files)
        ]
        files.append(("Text", ("", message)))
        try:
            r = requests.post(
                self.url + "/" + logbook,
                data=data,
                files=files,
                allow_redirects=False,
                verify=False,
            )
        except requests.RequestException as e:
            self.log.error("Cannot reach the ELOG server!\nError: {}".format(e))
        else:
            if r.status_code not in [200, 302]:
                self.log.error(
                    "Something went wrong...\n\nHere is what we got:\n{}".format(
                        r.content.decode("utf-8", "ignore")
                    )
                )
            else:
                self.cprint("ELOG post created successfully.")

        for (_, (_, fobj)) in files:
            if fobj and hasattr(fobj, "close"):
                fobj.close()


def create_elog_password_hash(password):
    """Create a SHA256 encrypted password for ELOGs."""
    from passlib.hash import sha256_crypt

    return sha256_crypt.encrypt(password, salt="", rounds=5000)[4:]
