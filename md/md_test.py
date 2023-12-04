from MVPython import  MarvelousDesignerAPI
from MVPython.MarvelousDesignerAPI import *
import MarvelousDesigner
from MarvelousDesigner import *


print("Module imported")

# import sys; sys.path.append("C:/project/HOOD/md"); import md_test; b = md_test.example(mdsa)

#If you want to load your python file, please type "sys.path.append(input path where file is located here)" in console.
class example():
    def __init__(self, mdsa):
        self.mdsa = mdsa
        
    def test(self):
        mdsa = self.mdsa
        mdsa.clear_console()
        #initialize mdsa module
        mdsa.initialize()
        print("initialize")
        
        #Set importing options (unit) of string type
        mdsa.set_import_scale_unit("m")
        
        #Set exporting options (unit) of string type
        mdsa.set_export_scale_unit("m")

        # Set simulation property settings
        # Set Simulation property options(simulation quality) of integer type
        # qulity = 0 Complete
        # qulity = 1 Normal
        # qulity = 2 Custom
        mdsa.set_simulation_quality(2)
        # Set Simulation property options(simulation time step) of floating point type
        mdsa.set_simulation_time_step(1/30)
        # Set Simulation property options(number of simulation) of integer type
        mdsa.set_simulation_number_of_simulation(1)
        # Set Simulation property options(simulation cg finish condition type) of integer type
        # cg finish condition type = 0 ITERATION
        # cg finish condition type = 1 RESIDUAL
        mdsa.set_simulation_cg_finish_condition(0)
        # Set Simulation property options(simulation cg iteration count) of integer type
        mdsa.set_simulation_cg_iteration_count(40)
        # Set Simulation property options(simulation cg residual) of floating point type
        # mdsa.on_simulation_cg_residual(0.00020)
        # Set Simulation property options(self collision iteration count) of integer type
        mdsa.set_simulation_self_collision_iteration_count(2)
        # Set Simulation property options(air damping) of floating point type
        mdsa.set_simulation_air_damping(1.0)
        # Set Simulation property options(gravity) of floating point type
        mdsa.set_simulation_gravity(-9800.00)
        # Set Simulation property options(number of CPU in use) of integer type
        mdsa.set_simulation_number_of_cpu_in_use(12)
        # Set Simulation property options(nonlinear simulation) of boolean type
        mdsa.set_simulation_nonlinear_simulation(False)
        # Set Simulation property options(ground collision) of boolean type
        mdsa.set_simulation_ground_collision(False)
        # Set Simulation property options(ground height) of floating point type
        mdsa.set_simulation_ground_height(-2.0)
        # Set Simulation property options(avatar-cloth collision detection triangle-vertex) of boolean type
        mdsa.set_simulation_avatar_cloth_collision_detection_triangle_vertex(False)
        # Set Simulation property options(self collision detection triangle-vertex) of boolean type
        mdsa.set_simulation_self_collision_detection_trianlge_vertex(False)
        # Set Simulation property options(self collision detection edge-edge) of boolean type
        mdsa.set_simulation_self_collision_detection_edge_edge(False)
        # Set Simulation property options(self collision detection avoidance stiffness) of floating point type
        mdsa.set_simulation_self_collision_detection_avoidance_stiffness(0.001111)
        # Set Simulation property options(proximity detection vertex-triangle) of boolean type
        mdsa.set_simulation_proximity_detection_vertex_triangle(False)
        # Set Simulation property options(proximity detection edge-edge) of boolean type
        mdsa.set_simulation_proximity_detection_edge_edge(False)
        # Set Simulation property options(intersection resolution) of boolean type
        mdsa.set_simulation_intersection_resolution(False)
        # Set Simulation property options(layer based collision detection) of boolean type
        mdsa.set_simulation_layer_based_collision_detection(False)
        
        
        #In case want to simulate/record one garment and avatar with multiple animation

        #set path of one garment file
        mdsa.set_garment_file_path("C:/data/neural_cloth/garment/test/dress.obj")

        # set path of one avatar file
        mdsa.set_avatar_file_path("C:/data/neural_cloth/human_motion/stretch.obj")


        #set folder path of multiple animation folder and extension (file extension must be supported by Marvelous Designer)
        mdsa.set_animation_file_path("C:/data/neural_cloth/human_motion/stretch.pc2")
        #set save folder and extension (file extension must be supported by Marvelous Designer)
        mdsa.set_save_file_path("C:/data/neural_cloth/md_test/test.pc2")
        #set auto save option. True is save with Zprj File and Image File.
        mdsa.set_auto_save(False)
        #call the "process" function (to autosave project file, change factor to ture)
        print("start")
        mdsa.process()
        
    def run_single_process_second_example(self):
        mdsa = self.mdsa
        # clear console window
        mdsa.clear_console() 
        #initialize mdsa module
        mdsa.initialize() 
        #set exporting option
        mdsa.set_open_option("cm", 30)
        #set importing option
        mdsa.set_save_option("mm", 30, False) 
        #designate the folder where the files will be stored and file extension setting
        mdsa.set_save_folder_path( "C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_4.2.281\\", "mc")
        #call the "single_process" function
        mdsa.single_process( "C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_4.2.281\\Garment\\Bag1.zpac", "C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_4.2.281\\Avatar\\Avatar\\Female_A_V3.avt", "C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_4.2.281\\Avatar\\Pose\\Female_A\\F_A_pose_02_attention.pos")

    def run_single_process_example(self):
        mdsa = self.mdsa
        # clear console window
        mdsa.clear_console()
        #initialize mdsa module
        mdsa.initialize()
        #set exporting option
        mdsa.set_open_option("mm", 30)
        #set importing option
        mdsa.set_save_option("mm", 30, False)
        #set Alembic Format True = Ogawa, False = hdf5. Default is hdf5. (This function is only valid when exporting alembic file.)
        #Set the path of an Avatar file you want to load.
        mdsa.set_avatar_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_4.2.281\\Avatar\\Avatar\\Female_A_V3.avt")
        
        mdsa.set_garment_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_4.2.281\\Garment\\Bag1.zpac")

        mdsa.set_animation_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_4.2.281\\Avatar\\Motion\\Female_A\\F_A_Type1_01_hands on waist_V0_1_2.mtn")

        #Set the saving file path.
        mdsa.set_save_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_4.2.281\\test_01.pc2")
        #set auto save option. True is save with Zprj File and Image File.
        mdsa.set_auto_save(True)
        #If you finish setting file paths and options. You must call process function.
        mdsa.process()