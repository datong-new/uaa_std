import torch
PWD = "/data/shudeng/attacks/"

MODEL_PATH = PWD + "models/"

IC15_TEST_IMAGES = "/data/shudeng/shudeng/IC15/test_images/"
TOTALTEXT_TEST_IMAGES = "/data/totaltext/totaltext/Images/Test/"

craft_res_dir = PWD + "res_craft/"
craft_universal_totaltext = craft_res_dir + "universal_totaltext/"
craft_single_totaltext = craft_res_dir + "single_totaltext/"
craft_universal_icdar = craft_res_dir + "universal_icdar/"
craft_single_icdar = craft_res_dir + "single_icdar/"

db_res_dir = PWD + "res_db/"
db_universal_totaltext = db_res_dir + "universal_totaltext/"
db_single_totaltext = db_res_dir + "single_totaltext/"

textbox_res_dir = PWD + "res_textbox/"
textbox_universal_icdar = textbox_res_dir + "universal_icdar/"
textbox_single_icdar = textbox_res_dir + "single_icdar/"

east_res_dir = PWD + "res_east/"
east_universal_icdar = east_res_dir + "universal_icdar/"
east_single_icdar = east_res_dir + "single_icdar/"

craft_var = db_var = textbox_var = torch.tensor([0.229, 0.224, 0.225]).mean().item()
east_var = 0.5


VARS = {"east":east_var , "db":db_var, "textbox":textbox_var, "craft":craft_var}
