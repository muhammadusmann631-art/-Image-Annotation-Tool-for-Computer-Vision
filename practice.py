class AdvancedImageAnnotator:
    def __init__(self): # constructor jonneche use krey gai
        self.drawing = False
        self.start_x = -1 #naye boxes drwa kai liye
        self.start_y = -1 # naye boxes draw kai liye
        self.current_x = -1 # current the
        self.current_y = -1 # current the
        self.boxes = [] # sare boxes draw idhr save
        self.current_class = 0 # jo naya hai 0 sai start ho ga 
        self.img = None # yahanimage
        self.img_copy = None # image copy none
        self.current_idx = 0 # current id 0
        self.images = [] # images save jo annotated ho kai aye hai
        self.save_dir = "" # jahan save krni hai 
        self.image_dir = "" # jahan image hai 
        self.img_size = (740, 700) # resize 
        
        # Edit mode features
        self.editing = True  # Default ON
        self.selected_box_index = -1 # selected boxes -1 matlab koi naye hai
        self.resizing = False # resize
        self.resizing_corner = None # corner ko resize image pai hi
        self.moving = False 
        self.offset_x = 0
        self.offset_y = 0
        self.handle_size = 8 # Handle size thoda kam kar diya for better precision
        
        # Template propagation
        self.template_boxes = [] # pechle images store krne kai liye 
        self.use_propagation = False  # boxes ko agli image pai copy krna False ki hia take na krna 
        