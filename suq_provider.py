# -*- coding: utf-8 -*-
"""
SUQProvider: QGIS Processing Provider for Slope Units for QGIS (SUQ).
"""

import os
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algorithms.suq_create_algorithm import SUQCreateAlgorithm
from .algorithms.suq_clean_algorithm import SUQCleanAlgorithm
from .algorithms.suq_metrics_algorithm import SUQMetricsAlgorithm
from .algorithms.suq_optimize_algorithm import SUQOptimizeAlgorithm
from .algorithms.suq_nested_algorithm import SUQNestedAlgorithm


class SUQProvider(QgsProcessingProvider):
    """Processing provider exposing the 5 SUQ slope unit algorithms."""

    def loadAlgorithms(self):
        """Register all SUQ algorithms."""
        self.addAlgorithm(SUQCreateAlgorithm())
        self.addAlgorithm(SUQCleanAlgorithm())
        self.addAlgorithm(SUQMetricsAlgorithm())
        self.addAlgorithm(SUQOptimizeAlgorithm())
        self.addAlgorithm(SUQNestedAlgorithm())

    def id(self):
        """Unique provider ID."""
        return "suq"

    def name(self):
        """Human-readable provider name shown in QGIS Processing Toolbox."""
        return "SUQ (Slope Units for QGIS)"

    def longName(self):
        """Extended provider name."""
        return "SUQ - Standalone Slope Units for QGIS (r.slopeunits v2.0+)"

    def icon(self):
        """Provider icon."""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()
