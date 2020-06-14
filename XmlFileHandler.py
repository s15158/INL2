import xml.etree.ElementTree as ET


class XmlFileHandler(object):
    def __init__(self):
        self.documents = list()
        self.tree = None
        self.root = None

    def _getDocuments(self):
        documentsBranch = self.root.find("documents")
        if documentsBranch:
            for document in documentsBranch:
                self.documents.append(document.text)

    def parse(self, filename):
        self.tree = ET.parse(f"{filename}.xml")
        self.root = self.tree.getroot()
        self._getDocuments()
