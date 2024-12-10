#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "bunny.hh"
#define TEST_OPENGL_ERROR()                                                             \
  do {		  							\
    GLenum err = glGetError(); 					                        \
    if (err != GL_NO_ERROR) std::cerr << "OpenGL ERROR! " << __LINE__ << " --- Err Code=" << err << std::endl;      \
  } while(0)


GLuint vao_id;
GLuint vao_bunny_id;
GLuint program_id;
GLuint program_bunny_id;
float anim_time;
std::vector<GLfloat> vertex_buffer_data_ground;

void anim() {
  GLint anim_time_location;
  glUseProgram(program_id);
  anim_time_location = glGetUniformLocation(program_id, "anim_time");TEST_OPENGL_ERROR();
  glUniform1f(anim_time_location, anim_time);TEST_OPENGL_ERROR();
  anim_time += 0.1;
  glutPostRedisplay();
}
void timer(int value) {
  anim();
  glutTimerFunc(33,timer,0);
}
void init_anim() {
  glutTimerFunc(33,timer,0);
}

void window_resize(int width, int height) {
  //std::cout << "glViewport(0,0,"<< width << "," << height << ");TEST_OPENGL_ERROR();" << std::endl;
  glViewport(0,0,width,height);TEST_OPENGL_ERROR();
}

void display() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);TEST_OPENGL_ERROR();
  glUseProgram(program_id);TEST_OPENGL_ERROR();
  glBindVertexArray(vao_id);TEST_OPENGL_ERROR();
  glDrawArrays(GL_POINTS, 0, vertex_buffer_data_ground.size());TEST_OPENGL_ERROR();
  glUseProgram(program_bunny_id);TEST_OPENGL_ERROR();
  glBindVertexArray(vao_bunny_id);TEST_OPENGL_ERROR();
  //glDrawArrays(GL_TRIANGLES, 0, vertex_buffer_data.size());TEST_OPENGL_ERROR();TEST_OPENGL_ERROR();
  glBindVertexArray(0);TEST_OPENGL_ERROR();
  glutSwapBuffers();
}

void init_glut(int &argc, char *argv[]) {
  //glewExperimental = GL_TRUE;
  glutInit(&argc, argv);
  glutInitContextVersion(4,5);
  glutInitContextProfile(GLUT_CORE_PROFILE | GLUT_DEBUG);
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);
  glutInitWindowSize(1024, 1024);
  glutInitWindowPosition ( 100, 100 );
  glutCreateWindow("Shader Programming");
  glutDisplayFunc(display);
  glutReshapeFunc(window_resize);
}

void init_ground_vector()
{
  for (float x = -1.0f; x <= 1.0f; x+=0.15f)
  {
    for (float z = -1.0f; z <= 1.0f; z+=0.16f)
    {
      vertex_buffer_data_ground.push_back(x);
      vertex_buffer_data_ground.push_back(0.0f);
       vertex_buffer_data_ground.push_back(z);
    }
  }
}

// void init_ground_vector()
// {
//   vertex_buffer_data_ground.push_back(0);
//   vertex_buffer_data_ground.push_back(0);
//   vertex_buffer_data_ground.push_back(0);
// }

bool init_glew() {
  if (glewInit()) {
    std::cerr << " Error while initializing glew";
    return false;
  }
  return true;
}

void init_GL() {
  glEnable(GL_DEPTH_TEST);TEST_OPENGL_ERROR();
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);TEST_OPENGL_ERROR();
  // glEnable(GL_CULL_FACE);TEST_OPENGL_ERROR();
  glClearColor(0.4,0.4,0.4,1.0);TEST_OPENGL_ERROR();
}


void init_object_vbo() {
  int max_nb_vbo = 5;
  int nb_vbo = 0;
  int index_vbo = 0;
  int nb_vbo_bunny = 0;
  int index_vbo_bunny = 0;
  GLuint vbo_ids[max_nb_vbo];
  GLuint vbo_bunny_ids[max_nb_vbo];

  GLint vertex_location = glGetAttribLocation(program_id,"position");TEST_OPENGL_ERROR();
  GLint vertex_bunny_location = glGetAttribLocation(program_bunny_id,"position");TEST_OPENGL_ERROR();
  GLint normal_smooth_bunny_location = glGetAttribLocation(program_bunny_id,"normalSmooth");TEST_OPENGL_ERROR();

  glGenVertexArrays(1, &vao_id);TEST_OPENGL_ERROR();
  glBindVertexArray(vao_id);TEST_OPENGL_ERROR();
  glGenVertexArrays(1, &vao_bunny_id);TEST_OPENGL_ERROR();

  if (vertex_location!=-1) nb_vbo++;
  if (vertex_bunny_location!=-1) nb_vbo_bunny++;
  if (normal_smooth_bunny_location!=-1) nb_vbo_bunny++;
  glGenBuffers(nb_vbo, vbo_ids);TEST_OPENGL_ERROR();
  glGenBuffers(nb_vbo_bunny, vbo_bunny_ids);TEST_OPENGL_ERROR();

  if (vertex_location!=-1) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[index_vbo++]);TEST_OPENGL_ERROR();
    glBufferData(GL_ARRAY_BUFFER, vertex_buffer_data_ground.size()*sizeof(float), vertex_buffer_data_ground.data(), GL_STATIC_DRAW);TEST_OPENGL_ERROR();
    glVertexAttribPointer(vertex_location, 3, GL_FLOAT, GL_FALSE, 0, 0);TEST_OPENGL_ERROR();
    glEnableVertexAttribArray(vertex_location);TEST_OPENGL_ERROR();
  }

  glBindVertexArray(vao_bunny_id);TEST_OPENGL_ERROR();

  if (vertex_bunny_location!=-1) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bunny_ids[index_vbo_bunny++]);TEST_OPENGL_ERROR();
    glBufferData(GL_ARRAY_BUFFER, vertex_buffer_data.size()*sizeof(float), vertex_buffer_data.data(), GL_STATIC_DRAW);TEST_OPENGL_ERROR();
    glVertexAttribPointer(vertex_bunny_location, 3, GL_FLOAT, GL_FALSE, 0, 0);TEST_OPENGL_ERROR();
    glEnableVertexAttribArray(vertex_bunny_location);TEST_OPENGL_ERROR();
  }

  if (normal_smooth_bunny_location!=-1) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bunny_ids[index_vbo_bunny++]);TEST_OPENGL_ERROR();
    glBufferData(GL_ARRAY_BUFFER, normal_smooth_buffer_data.size()*sizeof(float), normal_smooth_buffer_data.data(), GL_STATIC_DRAW);TEST_OPENGL_ERROR();
    glVertexAttribPointer(normal_smooth_bunny_location, 3, GL_FLOAT, GL_FALSE, 0, 0);TEST_OPENGL_ERROR();
    glEnableVertexAttribArray(normal_smooth_bunny_location);TEST_OPENGL_ERROR();
  }
  glBindVertexArray(0);
}


std::string load(const std::string &filename) {
  std::ifstream input_src_file(filename, std::ios::in);
  std::string ligne;
  std::string file_content="";
  if (input_src_file.fail()) {
    std::cerr << "FAIL\n";
    return "";
  }
  while(getline(input_src_file, ligne)) {
    file_content = file_content + ligne + "\n";
  }
  file_content += '\0';
  input_src_file.close();
  return file_content;
}

bool init_shaders() {
  std::string vertex_src = load("../vertex.vert");
  std::string geometry_src = load("../geometry.geom");
  std::string fragment_src = load("../fragment.frag");
  std::string vertex_bunny_src = load("../vertex_bunny.vert");
  std::string fragment_bunny_src = load("../fragment_bunny.frag");
  GLuint shader_id[3];
  GLuint shader_bunny_id[2];
  GLint compile_status = GL_TRUE;
  char *vertex_shd_src = (char*)std::malloc(vertex_src.length()*sizeof(char));
  char *geometry_shd_src = (char*)std::malloc(geometry_src.length()*sizeof(char));
  char *fragment_shd_src = (char*)std::malloc(fragment_src.length()*sizeof(char));
  char *vertex_bunny_shd_src = (char*)std::malloc(vertex_bunny_src.length()*sizeof(char));
  char *fragment_bunny_shd_src = (char*)std::malloc(fragment_bunny_src.length()*sizeof(char));
  vertex_src.copy(vertex_shd_src,vertex_src.length());
  geometry_src.copy(geometry_shd_src, geometry_src.length());
  fragment_src.copy(fragment_shd_src,fragment_src.length());
  vertex_bunny_src.copy(vertex_bunny_shd_src,vertex_bunny_src.length());
  fragment_bunny_src.copy(fragment_bunny_shd_src,fragment_bunny_src.length());


  shader_id[0] = glCreateShader(GL_VERTEX_SHADER);TEST_OPENGL_ERROR();
  shader_id[1] = glCreateShader(GL_GEOMETRY_SHADER);TEST_OPENGL_ERROR();
  shader_id[2] = glCreateShader(GL_FRAGMENT_SHADER);TEST_OPENGL_ERROR();
  shader_bunny_id[0] = glCreateShader(GL_VERTEX_SHADER);TEST_OPENGL_ERROR();
  shader_bunny_id[1] = glCreateShader(GL_FRAGMENT_SHADER);TEST_OPENGL_ERROR();

  glShaderSource(shader_id[0], 1, (const GLchar**)&(vertex_shd_src), 0);TEST_OPENGL_ERROR();
  glShaderSource(shader_id[1], 1, (const GLchar**)&(geometry_shd_src), 0);TEST_OPENGL_ERROR();
  glShaderSource(shader_id[2], 1, (const GLchar**)&(fragment_shd_src), 0);TEST_OPENGL_ERROR();
  glShaderSource(shader_bunny_id[0], 1, (const GLchar**)&(vertex_bunny_shd_src), 0);TEST_OPENGL_ERROR();
  glShaderSource(shader_bunny_id[1], 1, (const GLchar**)&(fragment_bunny_shd_src), 0);TEST_OPENGL_ERROR();
  for(int i = 0 ; i <= 2 ; i++) {
    glCompileShader(shader_id[i]);TEST_OPENGL_ERROR();
    glGetShaderiv(shader_id[i], GL_COMPILE_STATUS, &compile_status);
    if(compile_status != GL_TRUE) {
      GLint log_size;
      char *shader_log;
      glGetShaderiv(shader_id[i], GL_INFO_LOG_LENGTH, &log_size);
      shader_log = (char*)std::malloc(log_size+1); /* +1 pour le caractere de fin de chaine '\0' */
      if(shader_log != 0) {
	  glGetShaderInfoLog(shader_id[i], log_size, &log_size, shader_log);
	  std::cerr << "SHADER " << i << ": " << shader_log << std::endl;
	  std::free(shader_log);
      }
      std::free(vertex_shd_src);
      std::free(geometry_shd_src);
      std::free(fragment_shd_src);
      std::free(vertex_bunny_shd_src);
      std::free(fragment_bunny_shd_src);
      glDeleteShader(shader_id[0]);
      glDeleteShader(shader_id[1]);
      glDeleteShader(shader_id[2]);
      glDeleteShader(shader_bunny_id[0]);
      glDeleteShader(shader_bunny_id[1]);
      return false;
    }
  }


  for(int i = 0 ; i <= 1 ; i++) {
    glCompileShader(shader_bunny_id[i]);TEST_OPENGL_ERROR();
    glGetShaderiv(shader_bunny_id[i], GL_COMPILE_STATUS, &compile_status);
    if(compile_status != GL_TRUE) {
      GLint log_size;
      char *shader_log;
      glGetShaderiv(shader_bunny_id[i], GL_INFO_LOG_LENGTH, &log_size);
      shader_log = (char*)std::malloc(log_size+1); /* +1 pour le caractere de fin de chaine '\0' */
      if(shader_log != 0) {
        glGetShaderInfoLog(shader_bunny_id[i], log_size, &log_size, shader_log);
        std::cerr << "SHADER BUNNY" << i << ": " << shader_log << std::endl;
        std::free(shader_log);
      }
      std::free(vertex_shd_src);
      std::free(geometry_shd_src);
      std::free(fragment_shd_src);
      std::free(vertex_bunny_shd_src);
      std::free(fragment_bunny_shd_src);
      glDeleteShader(shader_id[0]);
      glDeleteShader(shader_id[1]);
      glDeleteShader(shader_id[2]);
      glDeleteShader(shader_bunny_id[0]);
      glDeleteShader(shader_bunny_id[1]);
      return false;
    }
  }

  std::free(vertex_shd_src);
  std::free(geometry_shd_src);
  std::free(fragment_shd_src);
  std::free(vertex_bunny_shd_src);
  std::free(fragment_bunny_shd_src);


  GLint link_status=GL_TRUE;
  program_id=glCreateProgram();TEST_OPENGL_ERROR();
  program_bunny_id=glCreateProgram();TEST_OPENGL_ERROR();
  if (program_id==0) return false;
  if (program_bunny_id == 0) return false;
  for(int i = 0 ; i <= 2 ; i++) {
    glAttachShader(program_id, shader_id[i]);TEST_OPENGL_ERROR();
    if (i != 2) glAttachShader(program_bunny_id, shader_bunny_id[i]);TEST_OPENGL_ERROR();
  }


  glLinkProgram(program_id);TEST_OPENGL_ERROR();
  glGetProgramiv(program_id, GL_LINK_STATUS, &link_status);
  if (link_status!=GL_TRUE) {
    GLint log_size;
    char *program_log;
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &log_size);
    program_log = (char*)std::malloc(log_size+1); /* +1 pour le caractere de fin de chaine '\0' */
    if(program_log != 0) {
      glGetProgramInfoLog(program_id, log_size, &log_size, program_log);
      std::cerr << "Program " << program_log << std::endl;
      std::free(program_log);
    }
    glDeleteProgram(program_id);TEST_OPENGL_ERROR();
    glDeleteProgram(program_bunny_id);TEST_OPENGL_ERROR();
    glDeleteShader(shader_id[0]);TEST_OPENGL_ERROR();
    glDeleteShader(shader_id[1]);TEST_OPENGL_ERROR();
    glDeleteShader(shader_id[2]);TEST_OPENGL_ERROR();
    glDeleteShader(shader_bunny_id[0]);TEST_OPENGL_ERROR();
    glDeleteShader(shader_bunny_id[1]);TEST_OPENGL_ERROR();
    program_id=0;
    program_bunny_id=0;
    return false;
  }

  glLinkProgram(program_bunny_id);TEST_OPENGL_ERROR();
  glGetProgramiv(program_bunny_id, GL_LINK_STATUS, &link_status);
  if (link_status!=GL_TRUE) {
    GLint log_size;
    char *program_log;
    glGetProgramiv(program_bunny_id, GL_INFO_LOG_LENGTH, &log_size);
    program_log = (char*)std::malloc(log_size+1); /* +1 pour le caractere de fin de chaine '\0' */
    if(program_log != 0) {
      glGetProgramInfoLog(program_bunny_id, log_size, &log_size, program_log);
      std::cerr << "Program bunny" << program_log << std::endl;
      std::free(program_log);
    }
    glDeleteProgram(program_id);TEST_OPENGL_ERROR();
    glDeleteProgram(program_bunny_id);TEST_OPENGL_ERROR();
    glDeleteShader(shader_id[0]);TEST_OPENGL_ERROR();
    glDeleteShader(shader_id[1]);TEST_OPENGL_ERROR();
    glDeleteShader(shader_id[2]);TEST_OPENGL_ERROR();
    glDeleteShader(shader_bunny_id[0]);TEST_OPENGL_ERROR();
    glDeleteShader(shader_bunny_id[1]);TEST_OPENGL_ERROR();
    program_id=0;
    program_bunny_id=0;
    return false;
  }
  glUseProgram(program_id);TEST_OPENGL_ERROR();
  return true;
}


int main(int argc, char *argv[]) {
  init_glut(argc, argv);
  if (!init_glew())
    std::exit(-1);
  init_GL();
  init_shaders();
  init_ground_vector();
  init_object_vbo();
  init_anim();
  glutMainLoop();
}
