#pragma once


#include "cgp/cgp.hpp"
#include "environment.hpp"
#include "simulation/simulation.hpp"


using cgp::mesh_drawable;
using cgp::curve_drawable;
using cgp::mesh;
using cgp::numarray;
using cgp::vec3;



struct gui_parameters {
	bool display_frame = true;
	bool display_wireframe = false;
};

// The structure of the custom scene
struct scene_structure : cgp::scene_inputs_generic {
	
	// ****************************** //
	// Elements and shapes of the scene
	// ****************************** //
	camera_controller_orbit_euler camera_control;
	camera_projection_perspective camera_projection;
	window_structure window;

	mesh_drawable global_frame;          // The standard global frame
	environment_structure environment;   // Standard environment controler
	input_devices inputs;                // Storage for inputs status (mouse, keyboard, window dimension)
	gui_parameters gui;                  // Standard GUI element storage
	
	// ****************************** //
	// Elements and shapes of the scene
	// ****************************** //

	//elevator control
	float target_angle = -43.6f;
	float current_angle = 0.0f;
	float rotation_speed = 30.0f;

	mesh_drawable elevator;
	mesh_drawable elevator_back;
	mesh_drawable back;
	mesh_drawable cylindre_centre;
	mesh_drawable elevator_left;
	mesh_drawable elevator_right;
	mesh_drawable floor;

	mesh_drawable cylindre_up0_0;
	mesh_drawable cylindre_up0_1;
	mesh_drawable cylindre_up0_2;
	mesh_drawable cylindre_up0_3;
	mesh_drawable cylindre_up0_4;
	mesh_drawable cylindre_up0_5;
	mesh_drawable cylindre_up0_6;
	mesh_drawable cylindre_up0_7;
	mesh_drawable cylindre_up0_8;

	mesh_drawable cylindre_up1_0;
	mesh_drawable cylindre_up1_1;
	mesh_drawable cylindre_up1_2;
	mesh_drawable cylindre_up1_3;
	mesh_drawable cylindre_up1_4;
	mesh_drawable cylindre_up1_5;
	mesh_drawable cylindre_up1_6;
	mesh_drawable cylindre_up1_7;

	mesh_drawable cylindre_up2_0;
	mesh_drawable cylindre_up2_1;
	mesh_drawable cylindre_up2_2;
	mesh_drawable cylindre_up2_3;
	//mesh_drawable cylindre_up2_4;
	mesh_drawable cylindre_up2_5;
	mesh_drawable cylindre_up2_6;
	mesh_drawable cylindre_up2_7;
	mesh_drawable cylindre_up2_8;

	mesh_drawable cylindre_mid_0;
	mesh_drawable cylindre_mid_1;
	mesh_drawable cylindre_mid_2;
	//mesh_drawable cylindre_mid_3;
	//mesh_drawable cylindre_mid_4;
	mesh_drawable cylindre_mid_5;
	mesh_drawable cylindre_mid_6;
	mesh_drawable cylindre_mid_7;

	mesh_drawable cylindre_down0_0;
	mesh_drawable cylindre_down0_1;
	mesh_drawable cylindre_down0_2;
	mesh_drawable cylindre_down0_3;
	mesh_drawable cylindre_down0_4;
	mesh_drawable cylindre_down0_5;
	mesh_drawable cylindre_down0_6;
	mesh_drawable cylindre_down0_7;
	mesh_drawable cylindre_down0_8;

	mesh_drawable cylindre_down1_0;
	mesh_drawable cylindre_down1_1;
	mesh_drawable cylindre_down1_2;
	mesh_drawable cylindre_down1_3;
	mesh_drawable cylindre_down1_4;
	mesh_drawable cylindre_down1_5;
	mesh_drawable cylindre_down1_6;
	mesh_drawable cylindre_down1_7;

	mesh_drawable cylindre_down2_0;
	mesh_drawable cylindre_down2_1;
	mesh_drawable cylindre_down2_2;
	mesh_drawable cylindre_down2_3;
	//mesh_drawable cylindre_down2_4;
	mesh_drawable cylindre_down2_5;
	mesh_drawable cylindre_down2_6;
	mesh_drawable cylindre_down2_7;
	mesh_drawable cylindre_down2_8;

	mesh_drawable basket0;
	mesh_drawable basket1;
	mesh_drawable basket2;
	mesh_drawable basket3;

	mesh_drawable score0;
	mesh_drawable score1;
	mesh_drawable score2;
	mesh_drawable score3;
	mesh_drawable score4;

	numarray<vec3> elevator_pos_init;
	numarray<vec3> all_elevator_pos;
	numarray<vec3> elevator_norm_init;
	numarray<vec3> all_elevator_norm;

	mesh_drawable sphere;
	std::vector<particle_structure> particles;

	timer_event_periodic timer;

	// ****************************** //
	// Functions
	// ****************************** //

	void initialize();    // Standard initialization to be called before the animation loop
	void display_frame(); // The frame display to be called within the animation loop
	void display_gui();   // The display of the GUI, also called within the animation loop


	void mouse_move_event();
	void mouse_click_event();
	void keyboard_event();
	void idle_frame();

	numarray<vec3> cylindre_fix_plan(const numarray<vec3>& pos);
	void sphere_display();
	void spawn_ball();

	float ELEVATOR_AMPLITUDE = 2.1f;
	float ELEVATOR_FREQUENCY = 0.05f;

	bool ELEVATOR_PAUSE = false;
	float ELEVATOR_PAUSE_START_TIME = 0.0f;
	float ELEVATOR_PAUSE_TEMPS = 1.0f; // en s

	bool INIT_TRANSPOSE = true;

};





