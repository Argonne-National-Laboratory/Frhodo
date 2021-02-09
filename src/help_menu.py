# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

from qtpy.QtWidgets import QMessageBox, QLabel
from qtpy import QtCore, QtGui
import requests, re


github_link = "https://github.com/Argonne-National-Laboratory/Frhodo/releases"

class HelpMenu:
    def __init__(self, parent):
        self.parent = parent

        self.update_message()   # Check for updates

        parent.actionUpdate_Frhodo.triggered.connect(lambda: self.update_message(show_no_update=True))
        parent.actionAbout.triggered.connect(lambda: self.showAboutBox())

    def newVersionExists(self, current_version):
        response = requests.get("https://api.github.com/repos/Argonne-National-Laboratory/Frhodo/releases/latest")
        github_release_name = response.json()["name"]
        github_release_version = re.findall(r'''(\d+(?:\.\d+)*)''', github_release_name)[0]

        current_version = current_version.split('.')
        github_version = github_release_version.split('.')

        current_version = current_version + ['0']*(len(github_version) - len(current_version))
        github_version = github_version + ['0']*(len(current_version) - len(github_version))

        for (v, git_v) in zip(current_version, github_version):
            if int(git_v) > int(v):
                return True
            elif int(git_v) < int(v):
                return False

        return False

    def update_message(self, show_no_update=False):
        parent = self.parent
        url ='https://github.com/Argonne-National-Laboratory/Frhodo/releases'

        try:    # this would break without internet otherwise
            new_version_exists = self.newVersionExists(parent.version)
        except:
            return

        if new_version_exists:
            text = f'Frhodo update available at:<br><a href=\"{url}"\>Frhodo Github</a>'
            msgBox = self.msgBox = QMessageBox(parent)
            msgBox.setWindowTitle('Frhodo')
            msgBox.setText(text)
            msgBox.setTextFormat(QtCore.Qt.RichText)

            for child in msgBox.children():
                if type(child) == QLabel:
                    if child.text() == text:
                        child.setToolTip(url)
                        child.setOpenExternalLinks(False)
                        child.linkActivated.connect(lambda: self.openUrl(url))

            msgBox.show()
        elif show_no_update:
            msgBox = self.msgBox = QMessageBox(parent)
            msgBox.setWindowTitle('Frhodo')
            msgBox.setText('Frhodo is using the most current version')
            msgBox.show()

    def openUrl(self, text_url):
        self.msgBox.close()
        url = QtCore.QUrl(text_url)
        if not QtGui.QDesktopServices.openUrl(url):
            QMessageBox.warning(self, 'Open Url', 'Could not open url')

    def showAboutBox(self):
        parent = self.parent
        url ='https://github.com/Argonne-National-Laboratory/Frhodo/'
        text = [f'Frhodo {parent.version}', 
                'Developed by Travis Sikes and Robert Tranter', '',
                f'Home:\t <a href=\"{url}"\>Frhodo Github</a>', '',
                'Copyright © 2020, UChicago Argonne, LLC', 'Licensed under BSD-3-Clause']
        text = '<br>'.join(text)
        msgBox = self.msgBox = QMessageBox(parent)
        msgBox.setWindowTitle('About Frhodo')
        msgBox.setText(text)
        msgBox.setTextFormat(QtCore.Qt.RichText)
        msgBox.show()