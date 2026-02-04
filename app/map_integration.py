#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 2023

@author: lefteris

@subject: map integration
"""

# Module 3: Map Integration
import folium
from folium.features import DivIcon
from folium.plugins import MarkerCluster



class MapIntegration:
    def __init__(self, coord_init, base_data):
        self.map = folium.Map(location=coord_init, zoom_start=12)
        self.base_data = base_data

    def plot_base_data(self):
        marker_cluster = MarkerCluster()
        self.map.add_child(marker_cluster)
        for i in range(len(self.base_data)):
            
            base_loc = [self.base_data[i]['coordinates_lat'],
                             self.base_data[i]['coordinates_long']]
            #circle = folium.Circle(plant_coord,radius=1000,
            #                       color='#d35400', fill=True).add_child(folium.Popup("plant_"+str(plant_loc.iloc[i,0])))
            marker = folium.map.Marker(
                base_loc,
                # Create an icon as a text label
                icon=DivIcon(
                    icon_size=(20,20),
                    icon_anchor=(0,0),
                    html='<div style="font-size: 12; color:#080808;"><b>%s</b></div>' % ("bs_"+str(self.base_data[i]['id'])+"_"+str(self.base_data[i]['base_type'])),
                    )
                )
            #site_map.add_child(circle)
            self.map.add_child(marker)
            
            marker_cl = folium.Marker(location = base_loc,
                                   icon = folium.Icon(color='red', icon_color="white"),
                                   popup = "em_"+str(self.base_data[i]['id'])+"_"+str(self.base_data[i]['base_type']))
            marker_cluster.add_child(marker_cl)



    def plot_user_data(self, user_data):
        marker_cluster = MarkerCluster()
        self.map.add_child(marker_cluster)
        for i in range(len(user_data)):
            
            emergency_loc = [user_data[i]['coordinates_lat'],
                             user_data[i]['coordinates_long']]
            #circle = folium.Circle(plant_coord,radius=1000,
            #                       color='#d35400', fill=True).add_child(folium.Popup("plant_"+str(plant_loc.iloc[i,0])))
            marker = folium.map.Marker(
                emergency_loc,
                # Create an icon as a text label
                icon=DivIcon(
                    icon_size=(20,20),
                    icon_anchor=(0,0),
                    html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % ("em_"+str(user_data[i]['id'])),
                    )
                )
            #site_map.add_child(circle)
            self.map.add_child(marker)
            
            marker_cl = folium.Marker(location = emergency_loc,
                                   icon = folium.Icon(color='white', icon_color="black"),
                                   popup = "em_"+str(user_data[i]['id']))
            marker_cluster.add_child(marker_cl)
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

