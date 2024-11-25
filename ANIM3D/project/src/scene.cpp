#include "scene.hpp"

#include <algorithm>
#include <limits>


using namespace cgp;

void scene_structure::spawn_ball()
{
	vec3 const v = vec3(0.0f, 0.0f, 0.0f);
	particle_structure particle;
	particle.p = { 0,0.1,0.0 };
	particle.r = 0.08f;
	particle.c = {0,0,255};
	particle.v = v;
	particle.m = 1.0f;

	particle.status.pl = particle_structure::status::SPAWN;
	particle.status.pb = particle_structure::status::BASKET0;

	particles.push_back(particle);
}

numarray<vec3> scene_structure::cylindre_fix_plan(const numarray<vec3>& pos)
{
	numarray<vec3> planes;
	for (int i = 0; i < pos.size() / 2; i++)
	{
		vec3 const D = pos[i];
		vec3 const B = pos[pos.size() / 2 + i];
		vec3 const A = {B.x, 1, B.z};
		float pente = (B.z - D.z) / (B.y - D.y);

		planes.push_back({A.x, 1, D.z + pente * (A.y - D.y)});
	}
	return planes;
}

mesh create_mesh_with_hole(const numarray<vec3>& circle, float len)
{
	mesh filled_mesh;
	int size_circle = circle.size();

	vec3 center = circle[size_circle / 2] + (circle[0] - circle[size_circle / 2]) / 2.0f;
	particle_structure::cylindre_down.push_back(center);

	numarray<vec3> plan;
	plan.push_back({center.x + len, center.y, center.z + len});
	plan.push_back({center.x + len, center.y, center.z});
	plan.push_back({center.x + len, center.y, center.z - len});
	plan.push_back({center.x, center.y, center.z - len});
	plan.push_back({center.x - len, center.y, center.z - len});
	plan.push_back({center.x - len, center.y, center.z});
	plan.push_back({center.x - len, center.y, center.z + len});
	plan.push_back({center.x, center.y, center.z + len});

	int size_plan = plan.size();

	filled_mesh.position = circle;
	filled_mesh.position.push_back(plan);


	for (int i = 0; i < size_circle; ++i) {
		int a = -1;
		float a_dist = std::numeric_limits<float>::max();
		int b = -1;
		float b_dist = std::numeric_limits<float>::max();
		const vec3 actue_points = circle[i];
		for (int outer = 0; outer < size_plan; ++outer)
		{
			vec3 points_outer = plan[outer];
			float dist = std::sqrt(std::pow(points_outer.x - actue_points.x, 2) + std::pow(points_outer.y - actue_points.y, 2) + std::pow(points_outer.z - actue_points.z, 2));
			if (a == -1)
			{
				a = outer;
				a_dist = dist;
			}
			else if (dist < a_dist)
			{
				b = a;
				a = outer;
				b_dist = a_dist;
				a_dist = dist;
			}
			else if (dist < b_dist)
			{
				b = outer;
				b_dist = dist;
			}
		}
		filled_mesh.connectivity.push_back({i, size_circle + a, size_circle + b});
	}

	filled_mesh.connectivity.push_back({ size_circle - 2, size_circle - 1, size_circle + 1});
	filled_mesh.connectivity.push_back({ ((size_circle) / 4) * 3 , ((size_circle) / 4) * 3 + 1, size_circle + 3});
	filled_mesh.connectivity.push_back({ ((size_circle) / 4) * 2 , ((size_circle) / 4) * 2 + 1, size_circle + 5});
	filled_mesh.connectivity.push_back({ ((size_circle) / 4) * 1 , ((size_circle) / 4) * 1 + 1, size_circle + 7});

	for (int i = 2; i < size_plan; i+=2)
		filled_mesh.connectivity.push_back({size_circle + i, size_circle + 1 + i, size_circle - 1 + i});
	filled_mesh.connectivity.push_back({size_circle, size_circle + 1, size_circle + size_plan -1});

	filled_mesh.position[size_circle + 0] = {2, 1, 2}; // oui, + 0 ne sert a rien, c'est juste pour l'allignement visuelle du code
	filled_mesh.position[size_circle + 1] = {2, 1, 0};
	filled_mesh.position[size_circle + 2] = {2, 1, -2.2};
	filled_mesh.position[size_circle + 3] = {0, 1, -2.2};
	filled_mesh.position[size_circle + 4] = {-2, 1, -2.2};
	filled_mesh.position[size_circle + 5] = {-2, 1, 0};
	filled_mesh.position[size_circle + 6] = {-2, 1, 2};
	filled_mesh.position[size_circle + 7] = {0, 1, 2};

	filled_mesh.fill_empty_field();

	return filled_mesh;
}


mesh mesh_cylindre_cut(const numarray<vec3>& circle, const numarray<vec3>& cylinder)
{

	mesh filled_mesh;
	filled_mesh.position = circle;
	for (int i = cylinder.size() / 2; i < cylinder.size(); ++i)
		filled_mesh.position.push_back(cylinder[i]);

	int size = circle.size();
	for (int i = 0; i < size - 1; ++i)
	{
		filled_mesh.connectivity.push_back({i, size + i, i + 1});
		filled_mesh.connectivity.push_back({ i + 1, size + i + 1, size + i});
	}

	filled_mesh.fill_empty_field();
	return filled_mesh;
}


void scene_structure::initialize()
{

	// Set the behavior of the camera and its initial position
	camera_control.initialize(inputs, window);
	camera_control.set_rotation_axis_z();
	camera_control.look_at({ 0.0f, -5.0f, 0.0f }, { 0,0,0 }, { 0,0.0,0 });

	// Create a visual frame representing the coordinate system
	global_frame.initialize_data_on_gpu(mesh_primitive_frame());

	particle_structure::cylindre_up.push_back({0, 0, 0});
	particle_structure::cylindre_centre_radius = 0.25;
	particle_structure::cylindre_all_radius = 0.10;

	// Creation des mesh primitive
	mesh mesh_elevator = mesh_primitive_quadrangle({-2, -0.25, -0.10}, {-2, 0.25, 0.10}, {2, 0.25, 0.10}, {2, -0.25, -0.10});
	mesh mesh_cylindre_centre = mesh_primitive_cylinder(particle_structure::cylindre_centre_radius, {0, 1.5, -0.50}, particle_structure::cylindre_up[0], 2, 50, false);
	mesh mesh_cylindre_all = mesh_primitive_cylinder(particle_structure::cylindre_all_radius, {0, 1, 0}, particle_structure::cylindre_up[0], 2, 50, true);

	mesh mesh_elevator_back = mesh_primitive_quadrangle({-2, 1.5, -2.2}, {2, 1.5, -2.2}, {2, 1.5, 2.2}, {-2, 1.5, 2.2});
	mesh mesh_elevator_left = mesh_primitive_quadrangle({-2, 1.5, -2.2}, {-2, 0, -2.2}, {-2, 0, 2.2}, {-2, 1.5, 2.2});
	mesh mesh_elevator_right = mesh_primitive_quadrangle({2, 1.5, -2.2}, {2, 0, -2.2}, {2, 0, 2.2}, {2, 1.5, 2.2});
	mesh mesh_floor = mesh_primitive_quadrangle({-2, 0, -2.2}, {2, 0, -2.2}, {2, 1.5, -2.2}, {-2, 1.5, -2.2});
	mesh mesh_basket = mesh_primitive_quadrangle({-1.2, 0, -2.2}, {-1.2, 0, -1.75}, {-1.2, 1, -1.75}, {-1.2, 1, -2.2});
	mesh mesh_score = mesh_primitive_quadrangle({-2, 0.999, -2.2}, {-1.2, 0.999, -2.2}, {-1.2, 0.99, -1.75}, {-2, 0.999, -1.75});

	// Creation des mesh complexe
	numarray<vec3> circle = cylindre_fix_plan(mesh_cylindre_centre.position);
	mesh mesh_back = create_mesh_with_hole(circle, 2.0f);
	mesh_cylindre_centre = mesh_cylindre_cut(circle, mesh_cylindre_centre.position);

	mesh_score.normal = -mesh_score.normal;

	// Enregistrement des norms / positions pour la simulation
	elevator_pos_init = mesh_elevator.position;
	elevator_norm_init = -mesh_elevator.normal;

	particle_structure::points_spawn.push_back({0, 1, 0}); // mur invisible d'attente elevateur
	particle_structure::points_spawn.push_back(mesh_cylindre_centre.position);

	particle_structure::norm_spawn.push_back({0, -1, 0}); // mur invisible d'attente elevateur
	particle_structure::norm_spawn.push_back(mesh_cylindre_centre.normal);

	particle_structure::norm_elevator.push_back(mesh_elevator_back.normal);
	particle_structure::norm_elevator.push_back({0, 1, 0}); // mur "back"
	particle_structure::norm_elevator.push_back({1, 0, 0}); // mur "left"
	particle_structure::norm_elevator.push_back({-1, 0, 0}); // mur "right"
	all_elevator_norm = particle_structure::norm_elevator;
	particle_structure::norm_elevator.push_back(-mesh_elevator.normal);

	particle_structure::points_elevator.push_back(mesh_elevator_back.position);
	particle_structure::points_elevator.push_back({0, 1, 0}); // mur "back"
	particle_structure::points_elevator.push_back({-2, 1, 0}); // mur "left"
	particle_structure::points_elevator.push_back({2, 1, 0}); // mur "right"

	all_elevator_pos = particle_structure::points_elevator;
	particle_structure::points_elevator.push_back(mesh_elevator.position);


	particle_structure::norm_out = -mesh_elevator.normal;
	particle_structure::points_out = mesh_elevator.position;
	vec3 add_elevator_up_pos = {0, 0, ELEVATOR_AMPLITUDE - 0.50}; // -0.50 car pente initiale de 0.2 + amplitude de 0.2 + 0.1 de dif a 0
	std::transform(particle_structure::points_out.begin(), particle_structure::points_out.end(), particle_structure::points_out.begin(),
			   [add_elevator_up_pos](const vec3& v) { return v + add_elevator_up_pos;});

	// Init data_on_gpu
	back.initialize_data_on_gpu(mesh_back);
	cylindre_centre.initialize_data_on_gpu(mesh_cylindre_centre);
	elevator.initialize_data_on_gpu(mesh_elevator);
	elevator_back.initialize_data_on_gpu(mesh_elevator_back);
	sphere.initialize_data_on_gpu(mesh_primitive_sphere());
	elevator_left.initialize_data_on_gpu(mesh_elevator_left);
	elevator_right.initialize_data_on_gpu(mesh_elevator_right);
	floor.initialize_data_on_gpu(mesh_floor);

	cylindre_up0_0.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up0_1.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up0_2.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up0_3.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up0_4.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up0_5.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up0_6.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up0_7.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up0_8.initialize_data_on_gpu(mesh_cylindre_all);

	cylindre_up1_0.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up1_1.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up1_2.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up1_3.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up1_4.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up1_5.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up1_6.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up1_7.initialize_data_on_gpu(mesh_cylindre_all);

	cylindre_up2_0.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up2_1.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up2_2.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up2_3.initialize_data_on_gpu(mesh_cylindre_all);
	//cylindre_up2_4.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up2_5.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up2_6.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up2_7.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_up2_8.initialize_data_on_gpu(mesh_cylindre_all);

	cylindre_mid_0.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_mid_1.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_mid_2.initialize_data_on_gpu(mesh_cylindre_all);
	//cylindre_mid_3.initialize_data_on_gpu(mesh_cylindre_all);
	//cylindre_mid_4.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_mid_5.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_mid_6.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_mid_7.initialize_data_on_gpu(mesh_cylindre_all);

	cylindre_down0_0.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down0_1.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down0_2.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down0_3.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down0_4.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down0_5.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down0_6.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down0_7.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down0_8.initialize_data_on_gpu(mesh_cylindre_all);

	cylindre_down1_0.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down1_1.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down1_2.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down1_3.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down1_4.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down1_5.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down1_6.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down1_7.initialize_data_on_gpu(mesh_cylindre_all);

	cylindre_down2_0.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down2_1.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down2_2.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down2_3.initialize_data_on_gpu(mesh_cylindre_all);
	//cylindre_down2_4.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down2_5.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down2_6.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down2_7.initialize_data_on_gpu(mesh_cylindre_all);
	cylindre_down2_8.initialize_data_on_gpu(mesh_cylindre_all);

	basket0.initialize_data_on_gpu(mesh_basket);
	basket1.initialize_data_on_gpu(mesh_basket);
	basket2.initialize_data_on_gpu(mesh_basket);
	basket3.initialize_data_on_gpu(mesh_basket);

	score0.initialize_data_on_gpu(mesh_score);
	score1.initialize_data_on_gpu(mesh_score);
	score2.initialize_data_on_gpu(mesh_score);
	score3.initialize_data_on_gpu(mesh_score);
	score4.initialize_data_on_gpu(mesh_score);

	// Set default color
	back.material.color = vec3(0, 255, 0);
	elevator.material.color = vec3(255, 255, 0);
	elevator_back.material.color = vec3(255, 0, 255);
	cylindre_centre.material.color = vec3(255, 0, 0);
	elevator_left.material.color = vec3(0, 255, 255);
	elevator_right.material.color = vec3(0, 255, 255);

	// Set default alpha
	elevator_left.material.alpha = 0.3;
	elevator_right.material.alpha = 0.3;

	sphere.texture.load_and_initialize_texture_2d_on_gpu(project::path + "assets/lama.jpg");
	score0.texture.load_and_initialize_texture_2d_on_gpu(project::path + "assets/42.png");
	score1.texture.load_and_initialize_texture_2d_on_gpu(project::path + "assets/72.png");
	score2.texture.load_and_initialize_texture_2d_on_gpu(project::path + "assets/114.png");
	score3.texture.load_and_initialize_texture_2d_on_gpu(project::path + "assets/72.png");
	score4.texture.load_and_initialize_texture_2d_on_gpu(project::path + "assets/42.png");

	// Spawn first ball
	spawn_ball();
}


void scene_structure::display_frame()
{
	// Set the light to the current position of the camera
	environment.light = camera_control.camera_model.position();

	if (gui.display_frame)
		draw(global_frame, environment);

	timer.update();
	float time = timer.t;

	// pour faire une rotation facillement, on calule par rapport a l'origine on transpose apres qui sera calculer lors du draw
	if (INIT_TRANSPOSE)
	{
		elevator.model.translation = {0, 1.25, 0.10};
		vec3 add_one_y = {0, 1, 0};

		numarray<vec3> transpos_list;
		transpos_list.push_back({-1.90, 0, 0});
		transpos_list.push_back({-1.425, 0, 0});
		transpos_list.push_back({-0.95, 0, 0});
		transpos_list.push_back({-0.475, 0, 0});
		transpos_list.push_back({0, 0, 0});
		transpos_list.push_back({0.475, 0, 0});
		transpos_list.push_back({0.95, 0, 0});
		transpos_list.push_back({1.425, 0, 0});
		transpos_list.push_back({1.90, 0, 0});

		transpos_list.push_back({-1.6625, 0, 0});
		transpos_list.push_back({-1.1875, 0, 0});
		transpos_list.push_back({-0.7125, 0, 0});
		transpos_list.push_back({-0.2375, 0, 0});
		transpos_list.push_back({0.2375, 0, 0});
		transpos_list.push_back({0.7125, 0, 0});
		transpos_list.push_back({1.1875, 0, 0});
		transpos_list.push_back({1.6625, 0, 0});


		//==============================================================================================================
		// UP 0
		//==============================================================================================================
		vec3 up0 = {0, 0, 1.5};
		cylindre_up0_0.model.translation = transpos_list[0] + up0;
		cylindre_up0_1.model.translation = transpos_list[1] + up0;
		cylindre_up0_2.model.translation = transpos_list[2] + up0;
		cylindre_up0_3.model.translation = transpos_list[3] + up0;
		cylindre_up0_4.model.translation = transpos_list[4] + up0;
		cylindre_up0_5.model.translation = transpos_list[5] + up0;
		cylindre_up0_6.model.translation = transpos_list[6] + up0;
		cylindre_up0_7.model.translation = transpos_list[7] + up0;
		cylindre_up0_8.model.translation = transpos_list[8] + up0;
		particle_structure::cylindre_up.push_back(transpos_list[0] + up0);
		particle_structure::cylindre_up.push_back(transpos_list[1] + up0);
		particle_structure::cylindre_up.push_back(transpos_list[2] + up0);
		particle_structure::cylindre_up.push_back(transpos_list[3] + up0);
		particle_structure::cylindre_up.push_back(transpos_list[4] + up0);
		particle_structure::cylindre_up.push_back(transpos_list[5] + up0);
		particle_structure::cylindre_up.push_back(transpos_list[6] + up0);
		particle_structure::cylindre_up.push_back(transpos_list[7] + up0);
		particle_structure::cylindre_up.push_back(transpos_list[8] + up0);
		particle_structure::cylindre_down.push_back(transpos_list[0] + up0 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[1] + up0 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[2] + up0 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[3] + up0 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[4] + up0 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[5] + up0 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[6] + up0 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[7] + up0 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[8] + up0 + add_one_y);


		//==============================================================================================================
		// UP 1
		//==============================================================================================================
		vec3 up1 = {0, 0, 1.0};
		cylindre_up1_0.model.translation = transpos_list[9] + up1;
		cylindre_up1_1.model.translation = transpos_list[10] + up1;
		cylindre_up1_2.model.translation = transpos_list[11] + up1;
		cylindre_up1_3.model.translation = transpos_list[12] + up1;
		cylindre_up1_4.model.translation = transpos_list[13] + up1;
		cylindre_up1_5.model.translation = transpos_list[14] + up1;
		cylindre_up1_6.model.translation = transpos_list[15] + up1;
		cylindre_up1_7.model.translation = transpos_list[16] + up1;
		particle_structure::cylindre_up.push_back(transpos_list[9] + up1);
		particle_structure::cylindre_up.push_back(transpos_list[10] + up1);
		particle_structure::cylindre_up.push_back(transpos_list[11] + up1);
		particle_structure::cylindre_up.push_back(transpos_list[12] + up1);
		particle_structure::cylindre_up.push_back(transpos_list[13] + up1);
		particle_structure::cylindre_up.push_back(transpos_list[14] + up1);
		particle_structure::cylindre_up.push_back(transpos_list[15] + up1);
		particle_structure::cylindre_up.push_back(transpos_list[16] + up1);
		particle_structure::cylindre_down.push_back(transpos_list[9] + up1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[10] + up1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[11] + up1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[12] + up1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[13] + up1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[14] + up1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[15] + up1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[16] + up1 + add_one_y);


		//==============================================================================================================
		// UP 2
		//==============================================================================================================
		vec3 up2 = {0, 0, 0.5};
		cylindre_up2_0.model.translation = transpos_list[0] + up2;
		cylindre_up2_1.model.translation = transpos_list[1] + up2;
		cylindre_up2_2.model.translation = transpos_list[2] + up2;
		cylindre_up2_3.model.translation = transpos_list[3] + up2;
		//cylindre_up2_4.model.translation = transpos_list[4] + up2;
		cylindre_up2_5.model.translation = transpos_list[5] + up2;
		cylindre_up2_6.model.translation = transpos_list[6] + up2;
		cylindre_up2_7.model.translation = transpos_list[7] + up2;
		cylindre_up2_8.model.translation = transpos_list[8] + up2;
		particle_structure::cylindre_up.push_back(transpos_list[0] + up2);
		particle_structure::cylindre_up.push_back(transpos_list[1] + up2);
		particle_structure::cylindre_up.push_back(transpos_list[2] + up2);
		particle_structure::cylindre_up.push_back(transpos_list[3] + up2);
		// particle_structure::cylindre_up.push_back(transpos_list[4] + up2);
		particle_structure::cylindre_up.push_back(transpos_list[5] + up2);
		particle_structure::cylindre_up.push_back(transpos_list[6] + up2);
		particle_structure::cylindre_up.push_back(transpos_list[7] + up2);
		particle_structure::cylindre_up.push_back(transpos_list[8] + up2);
		particle_structure::cylindre_down.push_back(transpos_list[0] + up2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[1] + up2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[2] + up2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[3] + up2 + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[4] + up2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[5] + up2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[6] + up2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[7] + up2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[8] + up2 + add_one_y);


		//==============================================================================================================
		// MID
		//==============================================================================================================
		cylindre_mid_0.model.translation = transpos_list[9];
		cylindre_mid_1.model.translation = transpos_list[10];
		cylindre_mid_2.model.translation = transpos_list[11];
		//cylindre_mid_3.model.translation = transpos_list[12];
		//cylindre_mid_4.model.translation = transpos_list[13];
		cylindre_mid_5.model.translation = transpos_list[14];
		cylindre_mid_6.model.translation = transpos_list[15];
		cylindre_mid_7.model.translation = transpos_list[16];
		particle_structure::cylindre_up.push_back(transpos_list[9]);
		particle_structure::cylindre_up.push_back(transpos_list[10]);
		particle_structure::cylindre_up.push_back(transpos_list[11]);
		// particle_structure::cylindre_up.push_back(transpos_list[12]);
		// particle_structure::cylindre_up.push_back(transpos_list[13]);
		particle_structure::cylindre_up.push_back(transpos_list[14]);
		particle_structure::cylindre_up.push_back(transpos_list[15]);
		particle_structure::cylindre_up.push_back(transpos_list[16]);
		particle_structure::cylindre_down.push_back(transpos_list[9] + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[10] + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[11] + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[12] + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[13] + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[14] + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[15] + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[16] + add_one_y);


		//==============================================================================================================
		// DOWN 0
		//==============================================================================================================
		// vec3 down0 = {0, 0, -1.5};
		// cylindre_down0_0.model.translation = transpos_list[0] + down0;
		// cylindre_down0_1.model.translation = transpos_list[1] + down0;
		// cylindre_down0_2.model.translation = transpos_list[2] + down0;
		// cylindre_down0_3.model.translation = transpos_list[3] + down0;
		// cylindre_down0_4.model.translation = transpos_list[4] + down0;
		// cylindre_down0_5.model.translation = transpos_list[5] + down0;
		// cylindre_down0_6.model.translation = transpos_list[6] + down0;
		// cylindre_down0_7.model.translation = transpos_list[7] + down0;
		// cylindre_down0_8.model.translation = transpos_list[8] + down0;
		// particle_structure::cylindre_up.push_back(transpos_list[0] + down0);
		// particle_structure::cylindre_up.push_back(transpos_list[1] + down0);
		// particle_structure::cylindre_up.push_back(transpos_list[2] + down0);
		// particle_structure::cylindre_up.push_back(transpos_list[3] + down0);
		// particle_structure::cylindre_up.push_back(transpos_list[4] + down0);
		// particle_structure::cylindre_up.push_back(transpos_list[5] + down0);
		// particle_structure::cylindre_up.push_back(transpos_list[6] + down0);
		// particle_structure::cylindre_up.push_back(transpos_list[7] + down0);
		// particle_structure::cylindre_up.push_back(transpos_list[8] + down0);
		// particle_structure::cylindre_down.push_back(transpos_list[0] + down0 + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[1] + down0 + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[2] + down0 + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[3] + down0 + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[4] + down0 + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[5] + down0 + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[6] + down0 + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[7] + down0 + add_one_y);
		// particle_structure::cylindre_down.push_back(transpos_list[8] + down0 + add_one_y);


		//==============================================================================================================
		// DOWN 1
		//==============================================================================================================
		vec3 down1 = {0, 0, -1.0};
		cylindre_down1_0.model.translation = transpos_list[9] + down1;
		cylindre_down1_1.model.translation = transpos_list[10] + down1;
		cylindre_down1_2.model.translation = transpos_list[11] + down1;
		cylindre_down1_3.model.translation = transpos_list[12] + down1;
		cylindre_down1_4.model.translation = transpos_list[13] + down1;
		cylindre_down1_5.model.translation = transpos_list[14] + down1;
		cylindre_down1_6.model.translation = transpos_list[15] + down1;
		cylindre_down1_7.model.translation = transpos_list[16] + down1;
		particle_structure::cylindre_up.push_back(transpos_list[9] + down1);
		particle_structure::cylindre_up.push_back(transpos_list[10] + down1);
		particle_structure::cylindre_up.push_back(transpos_list[11] + down1);
		particle_structure::cylindre_up.push_back(transpos_list[12] + down1);
		particle_structure::cylindre_up.push_back(transpos_list[13] + down1);
		particle_structure::cylindre_up.push_back(transpos_list[14] + down1);
		particle_structure::cylindre_up.push_back(transpos_list[15] + down1);
		particle_structure::cylindre_up.push_back(transpos_list[16] + down1);
		particle_structure::cylindre_down.push_back(transpos_list[9] + down1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[10] + down1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[11] + down1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[12] + down1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[13] + down1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[14] + down1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[15] + down1 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[16] + down1 + add_one_y);


		//==============================================================================================================
		// DOWN 2
		//==============================================================================================================
		vec3 down2 = {0, 0, -0.5};
		cylindre_down2_0.model.translation = transpos_list[0] + down2;
		cylindre_down2_1.model.translation = transpos_list[1] + down2;
		cylindre_down2_2.model.translation = transpos_list[2] + down2;
		cylindre_down2_3.model.translation = transpos_list[3] + down2;
		//cylindre_down2_4.model.translation = transpos_list[4] + down2;
		cylindre_down2_5.model.translation = transpos_list[5] + down2;
		cylindre_down2_6.model.translation = transpos_list[6] + down2;
		cylindre_down2_7.model.translation = transpos_list[7] + down2;
		cylindre_down2_8.model.translation = transpos_list[8] + down2;
		particle_structure::cylindre_up.push_back(transpos_list[0] + down2);
		particle_structure::cylindre_up.push_back(transpos_list[1] + down2);
		particle_structure::cylindre_up.push_back(transpos_list[2] + down2);
		particle_structure::cylindre_up.push_back(transpos_list[3] + down2);
		particle_structure::cylindre_up.push_back(transpos_list[4] + down2);
		particle_structure::cylindre_up.push_back(transpos_list[5] + down2);
		particle_structure::cylindre_up.push_back(transpos_list[6] + down2);
		particle_structure::cylindre_up.push_back(transpos_list[7] + down2);
		particle_structure::cylindre_up.push_back(transpos_list[8] + down2);
		particle_structure::cylindre_down.push_back(transpos_list[0] + down2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[1] + down2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[2] + down2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[3] + down2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[4] + down2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[5] + down2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[6] + down2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[7] + down2 + add_one_y);
		particle_structure::cylindre_down.push_back(transpos_list[8] + down2 + add_one_y);


		//==============================================================================================================
		// BASKET
		//==============================================================================================================
		basket1.model.translation = {0.8, 0, 0};
		basket2.model.translation = {0.8 * 2, 0, 0};
		basket3.model.translation = {0.8 * 3, 0, 0};

		score1.model.translation = {0.8, 0, 0};
		score2.model.translation = {0.8 * 2, 0, 0};
		score3.model.translation = {0.8 * 3, 0, 0};
		score4.model.translation = {0.8 * 4, 0, 0};

	}
	INIT_TRANSPOSE = false;


	// Calcul rotation elevateur
	if (elevator.model.translation.z <= -0.5f)
	{
		if (current_angle > target_angle) {
			current_angle -= rotation_speed * timer.scale * 0.01f;
			if (current_angle < target_angle)
				current_angle = target_angle;
		}
	}
	else if (elevator.model.translation.z >= 0.5f)
	{
		if (current_angle < 0.0f) { // retour a l'angle initial 0.
			current_angle += rotation_speed * timer.scale * 0.01f;
			if (current_angle > 0.0f)
				current_angle = 0.0f;
		}
	}

	elevator.model.rotation = rotation_transform::from_axis_angle({1, 0, 0}, current_angle * Pi / 180.0f);


	// Calcul transposition de l'elevateur
	if (ELEVATOR_PAUSE)
	{
		if (time - ELEVATOR_PAUSE_START_TIME >= ELEVATOR_PAUSE_TEMPS)
			ELEVATOR_PAUSE = false;
	}
	else
	{
		float new_z_position = ELEVATOR_AMPLITUDE * std::sin(2 * 3.14159f * ELEVATOR_FREQUENCY * time);
		if (std::abs(new_z_position - ELEVATOR_AMPLITUDE) < 0.01f || std::abs(new_z_position + ELEVATOR_AMPLITUDE) < 0.01f)
		{
			ELEVATOR_PAUSE = true;
			ELEVATOR_PAUSE_START_TIME = time;
		}
		elevator.model.translation = {elevator.model.translation.x, elevator.model.translation.y, new_z_position};
	}

	// Update Elevator param
	particle_structure::points_elevator = all_elevator_pos;
	particle_structure::norm_elevator = all_elevator_norm;

	for (int i = 0; i < elevator_pos_init.size(); i++)
	{
		vec3 rotation_pos = elevator.model.rotation * elevator_pos_init[i];
		vec3 rotation_norm = elevator.model.rotation * elevator_norm_init[i];

		particle_structure::points_elevator.push_back(rotation_pos + elevator.model.translation);
		particle_structure::norm_elevator.push_back(rotation_norm);
	}


	// Mur invisible pour bloquer l'entrer de l'evelateur si besoin
	if (elevator.model.translation.z >= -0.5)
	{
		particle_structure::norm_spawn[0] = {0, -1, 0};
		particle_structure::points_spawn[0] = {0, 1, 0};
	}
	else
	{
		particle_structure::norm_spawn[0] = {0, 1, 0};
		particle_structure::points_spawn[0] = {0, -1, 0};
	}

	draw(back, environment);
	draw(elevator, environment);
	draw(elevator_back, environment);
	draw(cylindre_centre, environment);
	draw(floor, environment);

	draw(cylindre_up0_0, environment);
	draw(cylindre_up0_1, environment);
	draw(cylindre_up0_2, environment);
	draw(cylindre_up0_3, environment);
	draw(cylindre_up0_4, environment);
	draw(cylindre_up0_5, environment);
	draw(cylindre_up0_6, environment);
	draw(cylindre_up0_7, environment);
	draw(cylindre_up0_8, environment);

	draw(cylindre_up1_0, environment);
	draw(cylindre_up1_1, environment);
	draw(cylindre_up1_2, environment);
	draw(cylindre_up1_3, environment);
	draw(cylindre_up1_4, environment);
	draw(cylindre_up1_5, environment);
	draw(cylindre_up1_6, environment);
	draw(cylindre_up1_7, environment);

	draw(cylindre_up2_0, environment);
	draw(cylindre_up2_1, environment);
	draw(cylindre_up2_2, environment);
	draw(cylindre_up2_3, environment);
	//draw(cylindre_up2_4, environment);
	draw(cylindre_up2_5, environment);
	draw(cylindre_up2_6, environment);
	draw(cylindre_up2_7, environment);
	draw(cylindre_up2_8, environment);

	draw(cylindre_mid_0, environment);
	draw(cylindre_mid_1, environment);
	draw(cylindre_mid_2, environment);
	//draw(cylindre_mid_3, environment);
	//draw(cylindre_mid_4, environment);
	draw(cylindre_mid_5, environment);
	draw(cylindre_mid_6, environment);
	draw(cylindre_mid_7, environment);

	// draw(cylindre_down0_0, environment);
	// draw(cylindre_down0_1, environment);
	// draw(cylindre_down0_2, environment);
	// draw(cylindre_down0_3, environment);
	// draw(cylindre_down0_4, environment);
	// draw(cylindre_down0_5, environment);
	// draw(cylindre_down0_6, environment);
	// draw(cylindre_down0_7, environment);
	// draw(cylindre_down0_8, environment);

	draw(cylindre_down1_0, environment);
	draw(cylindre_down1_1, environment);
	draw(cylindre_down1_2, environment);
	draw(cylindre_down1_3, environment);
	draw(cylindre_down1_4, environment);
	draw(cylindre_down1_5, environment);
	draw(cylindre_down1_6, environment);
	draw(cylindre_down1_7, environment);

	draw(cylindre_down2_0, environment);
	draw(cylindre_down2_1, environment);
	draw(cylindre_down2_2, environment);
	draw(cylindre_down2_3, environment);
	//draw(cylindre_down2_4, environment);
	draw(cylindre_down2_5, environment);
	draw(cylindre_down2_6, environment);
	draw(cylindre_down2_7, environment);
	draw(cylindre_down2_8, environment);

	draw(basket0, environment);
	draw(basket1, environment);
	draw(basket2, environment);
	draw(basket3, environment);

	draw(score0, environment);
	draw(score1, environment);
	draw(score2, environment);
	draw(score3, environment);
	draw(score4, environment);


	glEnable(GL_BLEND); //Alpha ON pour OpenGL
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_FALSE);

	draw(elevator_left, environment);
	draw(elevator_right, environment);

	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);

	float const dt = 0.01f * timer.scale;
	simulate(particles, dt);
	sphere_display();
}


void scene_structure::sphere_display()
{
	// Display the particles as spheres
	size_t const N = particles.size();
	for (size_t k = 0; k < N; ++k)
	{
		particle_structure const& particle = particles[k];
		sphere.model.translation = particle.p;
		sphere.model.scaling = particle.r;

		draw(sphere, environment);
	}
}


void scene_structure::display_gui()
{
	bool spawn = false;
	bool kill = false;
	ImGui::Checkbox("Frame", &gui.display_frame);
	spawn |= ImGui::Button("Spawn ball");
	if (spawn)
		spawn_ball();
	kill |= ImGui::Button("Kill");
	if (kill)
		particles.clear();
}


void scene_structure::mouse_move_event()
{
	if (!inputs.keyboard.shift)
		camera_control.action_mouse_move(environment.camera_view);
}
void scene_structure::mouse_click_event()
{
	camera_control.action_mouse_click(environment.camera_view);
}
void scene_structure::keyboard_event()
{
	camera_control.action_keyboard(environment.camera_view);
}
void scene_structure::idle_frame()
{
	camera_control.idle_frame(environment.camera_view);
}

