# -*- coding: utf-8 -*-
"""
SUQPlugin: Registers the SUQ Processing Provider in QGIS.
"""

from qgis.core import QgsApplication
from .suq_provider import SUQProvider


class SUQPlugin:
    """QGIS Plugin Implementation for SUQ (Slope Units for QGIS)."""

    def __init__(self, iface=None):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        """Register the SUQ processing provider."""
        self.provider = SUQProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Initialize GUI / Processing integration."""
        self.initProcessing()

    def unload(self):
        """Unregister the SUQ processing provider on plugin unload."""
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
