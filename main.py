import cv2
import numpy as np
import os
import random

class AdvancedImageAnnotator:
    def __init__(self): # constructor jonneche use krey gai
        self.drawing = False
        self.start_x = -1
        self.start_y = -1
        self.current_x = -1
        self.current_y = -1
        self.boxes = []
        self.current_class = 0
        self.img = None
        self.img_copy = None
        self.current_idx = 0
        self.images = []
        self.save_dir = ""
        self.image_dir = ""
        self.img_size = (740, 700)
        
        # Edit mode features
        self.editing = True  # Default ON
        self.selected_box_index = -1
        self.resizing = False
        self.resizing_corner = None
        self.moving = False
        self.offset_x = 0
        self.offset_y = 0
        self.handle_size = 8 # Handle size thoda kam kar diya for better precision
        
        # Template propagation
        self.template_boxes = []
        self.use_propagation = False  # boxes ko agli image pai copy krna False ki hia take na krna 
        
        # Class names
        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
    def get_color_for_class(self, class_id):
        # Deterministic color for each class
        random.seed(class_id * 50 + 1) # Added +1 to ensure random seed is non-zero
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    def mouse_callback(self, event, x, y, flags, param):
        if not self.editing:
            return
            
        # --- Handle Resizing/Moving Start Check ---
        if self.selected_box_index != -1 and not (self.drawing or self.resizing or self.moving):
            box_data = self.boxes[self.selected_box_index]
            x1, y1, x2, y2 = box_data['box']
            
            # Check corners for resize
            corners = {
                'tl': (x1, y1), 'tr': (x2, y1),
                'bl': (x1, y2), 'br': (x2, y2)
            }
            
            for corner_name, (cx, cy) in corners.items():
                if abs(x - cx) <= self.handle_size and abs(y - cy) <= self.handle_size:
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.resizing = True
                        self.resizing_corner = corner_name
                        return # Exit to avoid checking for moving or drawing
                    break
            else: # If no corner was hit, check for moving
                # Check if clicking inside box for moving
                if x1 < x < x2 and y1 < y < y2:
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.moving = True
                        self.offset_x = x - x1
                        self.offset_y = y - y1
                        return # Exit
        
        # --- Handle Resizing in Progress ---
        if self.resizing:
            if event == cv2.EVENT_MOUSEMOVE:
                box_data = self.boxes[self.selected_box_index]
                x1, y1, x2, y2 = box_data['box']
                
                # Use current x/y as the new corner point
                
                if self.resizing_corner == 'tl':
                    x1, y1 = x, y
                elif self.resizing_corner == 'tr':
                    x2, y1 = x, y
                elif self.resizing_corner == 'bl':
                    x1, y2 = x, y
                elif self.resizing_corner == 'br':
                    x2, y2 = x, y
                
                # Re-sort coordinates to ensure x1 < x2 and y1 < y2
                final_x1 = min(x1, x2)
                final_y1 = min(y1, y2)
                final_x2 = max(x1, x2)
                final_y2 = max(y1, y2)
                
                # Check minimum size (optional but recommended)
                if final_x2 - final_x1 > 5 and final_y2 - final_y1 > 5:
                    self.boxes[self.selected_box_index]['box'] = (final_x1, final_y1, final_x2, final_y2)
                
            elif event == cv2.EVENT_LBUTTONUP:
                self.resizing = False
                self.resizing_corner = None
            return # Stop further processing if resizing
        
        # --- Handle Moving in Progress ---
        if self.moving:
            if event == cv2.EVENT_MOUSEMOVE:
                box_data = self.boxes[self.selected_box_index]
                x1_old, y1_old, x2_old, y2_old = box_data['box']
                w = x2_old - x1_old
                h = y2_old - y1_old
                
                new_x1 = x - self.offset_x
                new_y1 = y - self.offset_y
                
                # Keep box within image boundaries (Optional: for robustness)
                H, W = self.img.shape[:2]
                new_x1 = max(0, min(new_x1, W - w))
                new_y1 = max(0, min(new_y1, H - h))

                self.boxes[self.selected_box_index]['box'] = (new_x1, new_y1, new_x1 + w, new_y1 + h)
                
            elif event == cv2.EVENT_LBUTTONUP:
                self.moving = False
            return # Stop further processing if moving
        
        # --- Handle New Box Drawing ---
        if not (self.resizing or self.moving): # Only allow drawing if not resizing/moving
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_x = x
                self.start_y = y
                self.current_x = x
                self.current_y = y
                self.selected_box_index = -1
                
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.current_x = x
                self.current_y = y
                
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                x1 = min(self.start_x, x)
                y1 = min(self.start_y, y)
                x2 = max(self.start_x, x)
                y2 = max(self.start_y, y)
                
                if x2 - x1 > 5 and y2 - y1 > 5:
                    self.boxes.append({
                        'class': self.current_class,
                        'box': (x1, y1, x2, y2)
                    })
                    print(f"Box added - Class {self.classes[self.current_class]}")
                    # Select the newly added box
                    self.selected_box_index = len(self.boxes) - 1
                self.current_x = -1
                self.current_y = -1
            
        # --- Select box with right click ---
        elif event == cv2.EVENT_RBUTTONDOWN and not (self.resizing or self.moving or self.drawing):
            self.selected_box_index = -1
            # Iterate backwards to select the topmost box (latest drawn)
            for i in range(len(self.boxes) - 1, -1, -1):
                box_data = self.boxes[i]
                x1, y1, x2, y2 = box_data['box']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_box_index = i
                    print(f"Selected box {i} -> Class {self.classes[box_data['class']]}")
                    break
    
    def draw_boxes(self):
        for i, box_data in enumerate(self.boxes):
            box = box_data['box']
            class_id = box_data['class']
            color = self.get_color_for_class(class_id)
            thickness = 3 if i == self.selected_box_index else 2
            
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            cv2.rectangle(self.img_copy, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(self.img_copy, self.classes[class_id], (x1 + 5, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw handles for selected box
            if i == self.selected_box_index:
                # Use a smaller, solid color for handles
                handle_color = (255, 255, 255) if sum(color) < 300 else (0, 0, 0)
                
                for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    cv2.circle(self.img_copy, (cx, cy), self.handle_size, handle_color, -1)
                    cv2.circle(self.img_copy, (cx, cy), self.handle_size, color, 1) # Outline
    
    def update_display(self):
        if self.img is None:
            return
            
        self.img_copy = self.img.copy()
        
        # Draw existing boxes
        self.draw_boxes()
        
        # Draw currently drawing box
        if self.drawing and self.current_x != -1 and self.current_y != -1:
            color = self.get_color_for_class(self.current_class)
            cv2.rectangle(self.img_copy, (self.start_x, self.start_y), 
                          (self.current_x, self.current_y), color, 2)
            cv2.putText(self.img_copy, self.classes[self.current_class], 
                        (self.start_x + 5, max(self.start_y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show image name and current class
        image_file_name = os.path.basename(self.images[self.current_idx])
        info_text = f"{image_file_name} | Class: {self.classes[self.current_class]} | Boxes: {len(self.boxes)}"
        # Check if the image has enough height for the info text at the bottom
        text_y = self.img_copy.shape[0] - 10 if self.img_copy.shape[0] > 30 else 15
        cv2.putText(self.img_copy, info_text, 
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add class panel if editing
        if self.editing:
            panel_height = 30 + len(self.classes) * 25
            # Create a panel below the image
            panel = np.zeros((panel_height, self.img_copy.shape[1], 3), dtype=np.uint8)
            panel[:, :] = (40, 40, 40)
            
            # Combine image and panel
            full_display = np.concatenate((self.img_copy, panel), axis=0)
            
            y_start = self.img_copy.shape[0] + 25
            
            for i in range(len(self.classes)):
                color = self.get_color_for_class(i)
                text = f"{i} - Class {self.classes[i]}"
                
                # Check if this class is the active class for drawing new boxes
                if i == self.current_class:
                    text += " [ACTIVE]"
                    cv2.rectangle(full_display, (5, y_start + i * 25 - 18), 
                                  (250, y_start + i * 25 + 5), color, 2)
                # Check if the selected box belongs to this class
                elif self.selected_box_index != -1 and self.boxes[self.selected_box_index]['class'] == i:
                    text += " [SELECTED]"
                    cv2.rectangle(full_display, (5, y_start + i * 25 - 18), 
                                  (250, y_start + i * 25 + 5), (0, 255, 255), 2) # Yellow for selected box class

                cv2.putText(full_display, text, (10, y_start + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imshow('Annotation Tool', full_display)
        else:
            cv2.imshow('Annotation Tool', self.img_copy)
    
    def save_annotations(self):
        if len(self.boxes) == 0:
            print("WARNING: Koi box nahi hai! Save nahi ho raha.")
            return False
        
        img_name = os.path.basename(self.images[self.current_idx])
        base_name = os.path.splitext(img_name)[0]
        
        h, w = self.img.shape[:2]
        
        # Save YOLO format labels (original size ke coordinates)
        label_path = os.path.join(self.save_dir, 'labels', base_name + '.txt')
        
        with open(label_path, 'w') as f:
            for box_data in self.boxes:
                box = box_data['box']
                class_id = box_data['class']
                
                # Normalize coordinates
                x1, y1, x2, y2 = box
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = abs(x2 - x1) / w
                height = abs(y2 - y1) / h
                
                # Sanity check for normalized values
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Resize and save image
        resized_img = cv2.resize(self.img, self.img_size)
        img_save_path = os.path.join(self.save_dir, 'images', base_name + '.jpg')
        cv2.imwrite(img_save_path, resized_img)
        
        # Save annotated image (with boxes drawn) - resized
        annotated_img = cv2.resize(self.img.copy(), self.img_size)
        scale_x = self.img_size[0] / w
        scale_y = self.img_size[1] / h
        
        for box_data in self.boxes:
            box = box_data['box']
            class_id = box_data['class']
            color = self.get_color_for_class(class_id)
            
            # Scale coordinates to resized image
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, self.classes[class_id], (x1 + 5, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        annotated_path = os.path.join(self.save_dir, 'annotated', base_name + '_annotated.jpg')
        cv2.imwrite(annotated_path, annotated_img)
        
        print(f"✓ Saved: {base_name} (image + label + annotated) | Boxes: {len(self.boxes)}")
        
        # Update template for propagation
        if self.use_propagation:
            # Important: Save the original, unscaled coordinates for propagation
            self.template_boxes = [{'class': b['class'], 'box': b['box']} for b in self.boxes]
        
        return True
    
    def load_image(self):
        # Reset any active mouse operations
        self.drawing = self.resizing = self.moving = False
        self.start_x = self.start_y = self.current_x = self.current_y = -1
        self.resizing_corner = None
        
        if self.current_idx >= len(self.images):
            print("\n✓ Saari images complete ho gayi!")
            return False
        
        img_path = self.images[self.current_idx]
        self.img = cv2.imread(img_path)
        
        if self.img is None:
            print(f"ERROR: Image load nahi hui: {img_path}")
            self.current_idx += 1
            return self.load_image()  # Try next image
        
        self.img_copy = self.img.copy()
        
        # Apply template if propagation is enabled AND this is not the first load
        if self.use_propagation and len(self.template_boxes) > 0:
            # Use original coordinates for consistency
            self.boxes = [{'class': b['class'], 'box': b['box']} for b in self.template_boxes]
            print("✓ Template boxes applied!")
        else:
            # Load existing labels if available (optional feature, not implemented here)
            self.boxes = []
        
        self.selected_box_index = -1
        print(f"\n--- Image {self.current_idx + 1}/{len(self.images)}: {os.path.basename(img_path)} ---")
        return True
    # D:\image\images
    def run(self):
        print("=" * 60)
        print("         adfvanced images tool")
        print("=" * 60)
        
        # Get inputs
        self.image_dir = input("\nImages folder path: ").strip()
        if not os.path.exists(self.image_dir):
            print("ERROR: Folder nahi mila!")
            return
            
        self.save_dir = input("Save folder path: ").strip()
        
        # Create directories
        os.makedirs(os.path.join(self.save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'annotated'), exist_ok=True)
        
        # Get image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']
        self.images = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
                       if os.path.splitext(f)[1] in extensions]
        self.images.sort()
        
        if len(self.images) == 0:
            print("ERROR: Koi images nahi milein!")
            return
        
        print(f"\n✓ Total {len(self.images)} images milein")
        
        # Resume option
        resume = input("kahan sai start karna hai? (y/n): ").strip().lower()
        if resume == 'y':
            try:
                start_num = int(input(f"Kis number se start karna hai? (1-{len(self.images)}): "))
                self.current_idx = max(0, min(start_num - 1, len(self.images) - 1))
            except ValueError:
                 print("Invalid number. Starting from image 1.")
                 self.current_idx = 0
            
            # Ask for template propagation
            prop = input("Pehli image ke boxes agli images pe apply karein? (y/n): ").strip().lower()
            self.use_propagation = (prop == 'y')
        
        # Ask for image resize dimensions
        resize_opt = input(f"Image resize karna hai? Current: {self.img_size} (y/n): ").strip().lower()
        if resize_opt == 'y':
            try:
                w = int(input("Width (e.g., 640): "))
                h = int(input("Height (e.g., 640): "))
                self.img_size = (w, h)
            except ValueError:
                print("Invalid size, using default 640x640")
        
        # Load first image
        if not self.load_image():
            return
        
        cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        print("\n" + "=" * 60)
        print("                      CONTROLS")
        print("=" * 60)
        print("  K             - Edit mode ON/OFF")
        print("  Mouse Drag    - draw new box")
        print("  Right Click   - select box")
        print("  Drag Corner   - resize selected box (FIXED)")
        print("  Drag Box      - move selected box")
        print("  0-9           - select class / Change selected box class")
        print("  S             - save current image + next")
        print("  N             - next image (without save)")
        print("  B             - previous image")
        print("  D             - delete selected box (FIXED)")
        print("  Backspace/8   - delete last box (FIXED)")
        print("  ESC/Q         - exit (FIXED)")
        print("=" * 60 + "\n")
        
        while True:
            self.update_display()
            key = cv2.waitKey(1)
            
            # Escape or 'q' for Exit
            if key == 27 or key == ord('q'): 
                confirm = input("\nExit karna hai? (y/n): ").strip().lower()
                if confirm == 'y':
                    print("Exiting...")
                    break
            
            # Toggle edit mode
            elif key == ord('k'):
                self.editing = not self.editing
                print(f"Edit mode: {'ON ✓' if self.editing else 'OFF ✗'}")
            
            # Save and next
            elif key == ord('s'):
                if self.save_annotations():
                    self.current_idx += 1
                    if not self.load_image():
                        break
            
            # Next image without save
            elif key == ord('n'):
                self.current_idx += 1
                if not self.load_image():
                    break
            
            # Previous image
            elif key == ord('b'):
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.use_propagation = False  # No propagation backward
                    self.load_image()
            
            # Delete selected box ('d')
            elif key == ord('d') and self.selected_box_index != -1:
                removed_class = self.classes[self.boxes[self.selected_box_index]['class']]
                self.boxes.pop(self.selected_box_index)
                print(f"Deleted box {self.selected_box_index} - Class {removed_class}")
                self.selected_box_index = -1
            
            # Backspace - delete last box (key code 8)
            elif key == 8:
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"Last box deleted - Class {self.classes[removed['class']]}")
                    self.selected_box_index = -1
            
            # Class selection (0-9)
            elif 48 <= key <= 57:
                class_num = key - 48
                if class_num < len(self.classes):
                    if self.selected_box_index != -1:
                        # Change class of selected box
                        old_class = self.classes[self.boxes[self.selected_box_index]['class']]
                        self.boxes[self.selected_box_index]['class'] = class_num
                        print(f"Box class changed: {old_class} → {self.classes[class_num]}")
                    else:
                        # Change current class for new boxes
                        self.current_class = class_num
                        print(f"Current class: {self.classes[class_num]}")
        
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("         annotation tool band ho gaya!")
        print("=" * 60)

if __name__ == "__main__":
    try:
        annotator = AdvancedImageAnnotator()
        annotator.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()