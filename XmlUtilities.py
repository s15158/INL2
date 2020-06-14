import xml.etree.ElementTree as ET

# This module defines basic functions for parsing .xml format files to string objects.

def parseXml(filename):
    tree = ET.parse(f"en\\{filename}.xml")
    root = tree.getroot()
    documentsBranch = root.find("documents")
    return [document.text for document in documentsBranch]
