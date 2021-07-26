from dataclasses import dataclass
import math
from typing import List

@dataclass
class FDDBAnnotation:
    major_axis_radius:float
    minor_axis_radius:float
    angle:float
    center_x:float
    center_y:float
    
    @property
    def center(self):
        return (self.center_x, self.center_y)
    
    @property
    def axes(self):
        #return (int(self.major_axis_radius*2), int(self.minor_axis_radius*2))
        return (self.major_axis_radius*2, self.minor_axis_radius*2)
    
    def cv2_ellipse_box(self):
        return (self.center, self.axes, math.degrees(self.angle))
    
    def cv2_rectangle_box(self):
        radians = self.angle
        radians90 = radians + math.pi / 2
        radiusX, radiusY = self.major_axis_radius, self.minor_axis_radius
        ux = radiusX * math.cos(radians)
        uy = radiusX * math.sin(radians)
        vx = radiusY * math.cos(radians90)
        vy = radiusY * math.sin(radians90)
        width = math.sqrt(ux * ux + vx * vx) * 2
        height = math.sqrt(uy * uy + vy * vy) * 2
        x1, y1 = int(self.center_x - (width / 2)), int(self.center_y - (height / 2))
        x2, y2 = int(self.center_x + (width / 2)), int(self.center_y + (height / 2))
        return (x1, y1), (x2, y2)

@dataclass
class FDDBExample:

    fold:int
    image_path:str
    annotations:List[FDDBAnnotation]
        
        
    
    