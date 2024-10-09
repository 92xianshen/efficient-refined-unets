class InferenceConfig():
    def __init__(self) -> None:
        # ->> 1. Hyperparameter test
        # ->> theta_beta = 0.03125
        # self.theta_alpha = 120.0 # 4
        # self.theta_alpha = 80.0 # 1
        # self.theta_alpha = 40.0 # 2
        # self.theta_alpha = 10.0 # 3

        # ->> theta_alpha = 80.0
        # self.theta_beta = 0.03125 # 1
        # self.theta_beta = 0.0625 # 2
        # self.theta_beta = 0.125 # 3
        # self.theta_beta = 0.25 # 4

        # ->> Constant
        # self.theta_gamma = 3.0

        # ->> 2. Exploration of multi-spectral features
        # self.n_bands = 7 
        # self.img_channel_list, self.vis_channel_list = [4, 3, 2], None # RGB
        # self.save_path = "../../result/l8/rgb/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma) # RGB
        # self.img_channel_list, self.vis_channel_list = list(range(self.n_bands)), [4, 3, 2] # Seven-band
        # self.save_path = "../../result/l8/fullband/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma)

        # ->> 3. Ablation study with respect to the bilateral message-passing step
        # theta_alpha, theta_beta, theta_gamma = 80, .03125, 3
        
        # self.n_bands = 7 
        # self.img_channel_list, self.vis_channel_list = [4, 3, 2], None # RGB
        # self.save_path = "../../result/l8/wobilateral/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma)

        # # - Check model
        # self.theta_alpha = 80.0 
        # self.theta_beta = 0.03125
        # self.theta_gamma = 3.0
        # self.save_path = "../../output/l8/rgb/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma) # RGB

        # - 1. Hyper-parameter test
        self.theta_alpha = 80.0 # 80.0, 40.0, 10.0, 120.0 with theta_alpha=0.03125
        self.theta_beta = 0.03125 # 0.03125, 0.0625, 0.125, 0.25 with theta_alpha=80.0
        self.theta_gamma = 3.0
        
        # - Output path
        self.save_path = "output/l8/rgb/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma) # RGB
        # - Input and output
        self.ugenerator_path = "parameter/saved_model/unary_generator"
        self.data_path = "input/l8/"
        self.save_info_fname = "rfn.csv"
        
        # - Input and output shapes
        self.crop_height = 512
        self.crop_width = 512
        self.num_bands = 7
        self.num_classes = 4
    
        # - CRF parameter
        self.d_bifeats = 5
        self.d_spfeats = 2
        self.bilateral_compat = 10.0
        self.spatial_compat = 3.0
        self.num_iterations = 10