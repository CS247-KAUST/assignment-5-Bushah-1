#ifndef CS247_PROG_H
#define CS247_PROG_H

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <algorithm>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// framework includes
#include "glslprogram.h"
#include "vboquad.h"

////////////////
// Window size //
////////////////
const unsigned int gWindowWidth = 512;
const unsigned int gWindowHeight = 512;

//////////////////////
// Global variables //
//////////////////////

static GLFWwindow* window;

int current_scalar_field;
int data_size;
bool en_arrow;
bool en_streamline;
bool en_pathline;

int sampling_rate;
double dt;

char bmModifiers;
int  clearColor;

char* filenames[3];
bool  grid_data_loaded;
bool  scalar_data_loaded;
unsigned short vol_dim[3];
float* vector_array;
float* scalar_fields;
float* scalar_bounds;

GLuint scalar_field_texture;

int num_scalar_fields;
int num_timesteps;
int loaded_file;
int loaded_timestep;
float timestep;

int view_width, view_height;
int toggle_xy;

////////////////
// Prototypes //
////////////////

void drawGlyphs();
void computeStreamline(int x, int y);
void computePathline(int x, int y, int t);
void loadNextTimestep(void);
void LoadData(char* base_filename);
void LoadVectorData(const char* filename);
void DownloadScalarFieldAsTexture(void);
void reset_rendering_props(void);
bool initApplication(int argc, char** argv);

// Rendering objects
VBOQuad       quad;
GLSLProgram   vectorProgram;
glm::mat4     model;

#endif // CS247_PROG_H
