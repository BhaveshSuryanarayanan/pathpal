import numpy as np
import cv2


def draw_grid_on_image(image):
    
    HEIGHT,WIDTH,CHANNELS = image.shape

    
    one_third_height = HEIGHT//3
    one_third_width = WIDTH//3
    
    new_image = np.copy(image)
    cv2.line(new_image,(one_third_width,0),(one_third_width,HEIGHT),(0,0,0),5)
    cv2.line(new_image,(2*one_third_width,0),(2*one_third_width,HEIGHT),(0,0,0),5)
    cv2.line(new_image,(0,one_third_height),(WIDTH,one_third_height),(0,0,0),5)
    cv2.line(new_image,(0,2*one_third_height),(WIDTH,2*one_third_height),(0,0,0),5)
    return new_image


class VibrationMatrix():
    
    @staticmethod  # enable direct import from module
    def draw_vibration_matrix(matrix, img_shape, type = 'square'):
        if type == 'round':
            return VibrationMatrix.round_matrix(matrix, img_shape)
        else :
            return VibrationMatrix.square_matrix(matrix, img_shape)
        
    @staticmethod   
    def round_matrix(matrix, img_shape):
        HEIGHT,WIDTH = img_shape
        horizontal_outer_padding = WIDTH//10
        vertical_outer_padding = HEIGHT//10
        
        one_third_height = (HEIGHT-2*vertical_outer_padding)//3
        one_third_width = (WIDTH-2*horizontal_outer_padding)//3
        
        
        output = np.full((HEIGHT, WIDTH, 3), (0, 0, 0), dtype=np.uint8)
        
        xpadding = one_third_width//10
        ypadding = one_third_height//10
        radius = (min(one_third_height, one_third_width)*8)//10
        inner_radius = (min(one_third_height, one_third_width)*6)//10
        
        for i in range(3):
            for j in range(3):
                startx = horizontal_outer_padding + one_third_width*i + xpadding
                endx = horizontal_outer_padding + one_third_width*(i+1) - xpadding
                starty = vertical_outer_padding+one_third_height*j + ypadding
                endy = vertical_outer_padding+one_third_height*(j+1) - ypadding
                centre = ((startx+endx)//2, (starty+endy)//2)
                
                clr = (0,0,120) if matrix[j][i] else (50,50,50)
                clr2 = (0,0,200) if matrix[j][i] else (100,100,100)
                
                cv2.circle(output, centre,  radius//2, clr2, thickness=5)
                cv2.circle(output, centre , inner_radius//2, clr, thickness=-1)
        
        return output

    @staticmethod
    def square_matrix(matrix, img_shape):
        HEIGHT,WIDTH = img_shape
        horizontal_outer_padding = WIDTH//10
        vertical_outer_padding = HEIGHT//10
        
        one_third_height = (HEIGHT-2*vertical_outer_padding)//3
        one_third_width = (WIDTH-2*horizontal_outer_padding)//3
        
        
        
        output = np.full((HEIGHT, WIDTH, 3), (255, 255, 255), dtype=np.uint8)
        output[:] = [34,34,34]
        
        
        xpadding = one_third_width//10
        ypadding = one_third_height//10
        radius = one_third_height//5
        
        for i in range(3):
            for j in range(3):
                startx = horizontal_outer_padding + one_third_width*i + xpadding
                endx = horizontal_outer_padding + one_third_width*(i+1) - xpadding
                starty = vertical_outer_padding+one_third_height*j + ypadding
                endy = vertical_outer_padding+one_third_height*(j+1) - ypadding
                
                clr = (0,0,200) if matrix[j][i] else (0,200, 0)
                
                top_left, bottom_right = (startx, starty), (endx, endy)
                
                VibrationMatrix.draw_rounded_rectangle(output, top_left, bottom_right, radius, clr, thickness=cv2.FILLED)
                
        return output

    @staticmethod
    def draw_rounded_rectangle(img, top_left, bottom_right, radius, color, thickness=cv2.FILLED):
        """Draws a rounded rectangle by combining rectangles and ellipses."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        # Draw filled rectangle (excluding the rounded corners)
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

        # Draw four quarter-circle corners
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

        return img

def read_image_from_path(path):
    image = cv2.imread(path)
    return image