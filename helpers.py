import math

def on_track(coord, track):
    x,y = coord
    if(track == 2):
        if(x>50 and x<590 and y>50 and y<350):
            if(x < 350 and y > 230):
                #left
                return False
            else:
                if(x > 410 and x < 530 and y > 110 and y < 290):
                    #right
                    return False
                else:
                    if(x > 110 and x < 530 and y > 110 and y < 170):
                        return False
                    else:
                        return True
        else:
            return False
    else:    
        if(x>50 and x<590 and y>50 and y<350):
            if(x>290 and x<350 and y>230 and y<350):
                #bottom
                return False
            else:
                if(x>110 and x<530 and y>110 and y<170):
                    #top
                    return False
                else:
                    if(x>110 and x<230 and y>110 and y<290):
                        #left
                        return False
                    else:
                        if(x>410 and x<530 and y>110 and y<290):
                            return False
                        else:
                            return True
        else:
            return False

def detect_h(point, orientation, segment):
    check = (segment[0][1] > point[1] and 1*orientation > 180) or (segment[0][1] < point[1] and 1*orientation < 180)
    if(check):
        return -1 
    else:
        m = math.tan(orientation*.0174533)
        if(m != 0):
            x_int = point[0]+((segment[0][1]-point[1])/m)
            if(x_int <= max(segment[0][0],segment[1][0]) and x_int >= min(segment[0][0],segment[1][0])):
                return math.sqrt((segment[0][1]-point[1])**2+(x_int-point[0])**2)
            else:
                return -1
        else:
            return -1
    
def detect_v(point,orientation,segment):
    if((segment[0][0] > point[0] and orientation > 90 and orientation < 270) or ((segment[0][0] < point[0] and (orientation < 90 or orientation > 270)))):
        return -1
    else:
        m = math.tan(orientation*.0174533)
        y_int = point[1]+m*(segment[0][0]-point[0])
        if(y_int <= max(segment[0][1],segment[1][1]) and y_int >= min(segment[0][1],segment[1][1])):
            return math.sqrt((segment[0][0]-point[0])**2+(y_int-point[1])**2)
        else:
            return -1
