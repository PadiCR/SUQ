# -*- coding: utf-8 -*-
"""
SUQ (Slope Units for QGIS)
A standalone QGIS Processing plugin for automatic slope unit delineation.
"""


def classFactory(iface):
    """Load SUQPlugin class from file suq_plugin.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .suq_plugin import SUQPlugin

    return SUQPlugin(iface)
