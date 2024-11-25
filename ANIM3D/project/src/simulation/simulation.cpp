#include "simulation.hpp"

using namespace cgp;

numarray<vec3> particle_structure::norm_spawn;
numarray<vec3> particle_structure::points_spawn;
numarray<vec3> particle_structure::norm_elevator;
numarray<vec3> particle_structure::points_elevator;
numarray<vec3> particle_structure::norm_out;
numarray<vec3> particle_structure::points_out;
numarray<vec3> particle_structure::cylindre_up;
numarray<vec3> particle_structure::cylindre_down;
float particle_structure::cylindre_centre_radius;
float particle_structure::cylindre_all_radius;

void simple_impact(particle_structure& particle, const float attenuation = 0.8f)
{
	numarray<vec3> norms;
	numarray<vec3> pos;
	switch (particle.status.pl)
	{
	case (particle_structure::status::SPAWN):
		norms = particle_structure::norm_spawn;
		pos = particle_structure::points_spawn;
		break;
	case (particle_structure::status::ELEVATOR):
		norms = particle_structure::norm_elevator;
		pos = particle_structure::points_elevator;
		break;
	case (particle_structure::status::ELEVATOR_OUT):
		norms = particle_structure::norm_out;
		pos = particle_structure::points_out;
		break;
	case (particle_structure::status::CHUTE):
		norms.push_back({0, -1, 0});
		norms.push_back({0, 1, 0});
		norms.push_back({-1, 0, 0});
		norms.push_back({1, 0, 0});
		pos.push_back({0, 1, 0});
		pos.push_back({0, 0, 0});
		pos.push_back({2, 0, 0});
		pos.push_back({-2, 0, 0});
		break;
	case (particle_structure::status::END):
		norms.push_back({0, -1, 0});
		norms.push_back({0, 1, 0});
		norms.push_back({0, 0, 1});
		norms.push_back({1, 0, 0});
		norms.push_back({-1, 0, 0});
		pos.push_back({0, 1, 0});
		pos.push_back({0, 0, 0});
		pos.push_back({0, 0, -2.2});
		switch (particle.status.pb)
		{
			case (particle_structure::status::BASKET0):
				pos.push_back({-2, 0, 0});
				pos.push_back({-1.2, 0, 0});
				break;
			case (particle_structure::status::BASKET1):
				pos.push_back({-1.2, 0, 0});
				pos.push_back({-0.4, 0, 0});
				break;
			case (particle_structure::status::BASKET2):
				pos.push_back({-0.4, 0, 0});
				pos.push_back({0.4, 0, 0});
				break;
			case (particle_structure::status::BASKET3):
				pos.push_back({0.4, 0, 0});
				pos.push_back({1.2, 0, 0});
				break;
			case (particle_structure::status::BASKET4):
				pos.push_back({1.2, 0, 0});
				pos.push_back({2, 0, 0});
				break;
			default:
				pos.push_back({0, 0, 0});
				pos.push_back({0, 0, 0});
			break;
		}
		pos.push_back({0, 0, 0});
		pos.push_back({0, 0, 0});
		break;
	default:
		norms = numarray<vec3>(0);
		pos = numarray<vec3>(0);
	}
	for (int i = 0; i < norms.size(); i++)
	{
		float detection = dot(particle.p - pos.at(i), norms.at(i));
		{
			if (detection <= particle.r)
			{
				vec3 v_a = dot(particle.v, norms.at(i)) * norms.at(i);
				vec3 v_b = particle.v - dot(particle.v, norms.at(i)) * norms.at(i);

				particle.v = v_b - attenuation * v_a;
				float d = particle.r - dot(particle.p - pos.at(i), norms.at(i));
				particle.p += d*norms.at(i);
				if (particle.status.pl == particle_structure::status::SPAWN)
					break;
			}
		}
	}
}

void cylindre_impact(particle_structure& particle)
{
	float radius = particle_structure::cylindre_centre_radius;
	for (int i = 0; i < particle_structure::cylindre_up.size(); i++)
	{
		vec3 up = particle_structure::cylindre_up[i];
		vec3 down = particle_structure::cylindre_down[i];
		vec3 cylinder_axis = up - down;
		vec3 particle_to_base = particle.p - down;

		float projection_length = dot(particle_to_base, cylinder_axis) / norm(cylinder_axis);
		vec3 projection_point = down + projection_length * normalize(cylinder_axis);

		vec3 radial_vector = particle.p - projection_point;
		float distance_to_axis = norm(radial_vector);

		if (distance_to_axis <= radius + particle.r)
		{
			vec3 collision_normal = normalize(radial_vector);

			vec3 v_parallel = dot(particle.v, collision_normal) * collision_normal;
			vec3 v_perpendicular = particle.v - v_parallel;

			particle.v = v_perpendicular - v_parallel;

			float overlap = (radius + particle.r) - distance_to_axis;
			particle.p += overlap * collision_normal;
		}
		else
		{
			simple_impact(particle);
		}
		radius = particle_structure::cylindre_all_radius;
	}
}

void particle_collisions(std::vector<particle_structure>& particles, float attenuation = 0.8f)
{
	size_t N = particles.size();
	for (size_t i = 0; i < N; ++i)
	{
		particle_structure& p1 = particles[i];
		for (size_t j = i + 1; j < N; ++j)
		{
			particle_structure& p2 = particles[j];

			vec3 diff = p2.p - p1.p;
			float dist = norm(diff);
			float min_dist = p1.r + p2.r;

			if (dist < min_dist)
			{
				vec3 collision_normal = normalize(diff);

				float v1_normal = dot(p1.v, collision_normal);
				float v2_normal = dot(p2.v, collision_normal);

				float m1 = p1.m;
				float m2 = p2.m;
				float v1_normal_after = (v1_normal * (m1 - attenuation * m2) + (1 + attenuation) * m2 * v2_normal) / (m1 + m2);
				float v2_normal_after = (v2_normal * (m2 - attenuation * m1) + (1 + attenuation) * m1 * v1_normal) / (m1 + m2);

				p1.v += (v1_normal_after - v1_normal) * collision_normal;
				p2.v += (v2_normal_after - v2_normal) * collision_normal;

				float overlap = min_dist - dist;
				vec3 correction = overlap * collision_normal / 2.0f;
				p1.p -= correction;
				p2.p += correction;
			}
		}
	}
}


void simulate(std::vector<particle_structure>& particles, float dt_arg)
{
	size_t const N_substep = 20;
	float const dt = dt_arg / N_substep;
	for (size_t k_substep = 0; k_substep < N_substep; ++k_substep)
	{

		vec3 const g = { 0,0,-9.81f };
		size_t const N = particles.size();


		for (size_t k = 0; k < N; ++k)
		{
			particle_structure& particle = particles[k];
			simple_impact(particle);
			if (particle.status.pl == particle_structure::status::CHUTE)
				cylindre_impact(particle);
		}

		particle_collisions(particles);

		// Update velocity with gravity force and friction
		for (size_t k = 0; k < N; ++k)
		{
			particle_structure& particle = particles[k];
			vec3 const f = particle.m * g;

			particle.v = (1 - 0.9f * dt) * particle.v + dt * f / particle.m;
		}

		// Update position from velocity
		for (size_t k = 0; k < N; ++k)
		{
			particle_structure& particle = particles[k];
			particle.p = particle.p + dt * particle.v;
			if (particle.status.pl == particle_structure::status::SPAWN && particle.p.y >= 1.10f )
				particle.status.pl = particle_structure::status::ELEVATOR;
			else if (particle.status.pl == particle_structure::status::ELEVATOR && particle.p.z >= 2.10f )
				particle.status.pl = particle_structure::status::ELEVATOR_OUT;
			else if (particle.status.pl == particle_structure::status::ELEVATOR_OUT && particle.p.y <= 0.95f)
				particle.status.pl = particle_structure::status::CHUTE;
			else if (particle.status.pl == particle_structure::status::CHUTE && particle.p.z < -1.75f)
			{
				particle.status.pl = particle_structure::status::END;
				float posX = particle.p.x;
				if (posX <= -1.2f)
					particle.status.pb = particle_structure::status::BASKET0;
				else if (posX <= -0.4f)
					particle.status.pb = particle_structure::status::BASKET1;
				else if (posX <= 0.4f)
					particle.status.pb = particle_structure::status::BASKET2;
				else if (posX <= 1.2f)
					particle.status.pb = particle_structure::status::BASKET3;
				else
					particle.status.pb = particle_structure::status::BASKET4;
			}
			else if (particle.status.pl == particle_structure::status::END && particle.p.z >= -1.75f)
				particle.status.pl = particle_structure::status::CHUTE;
		}
	}

}
