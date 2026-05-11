// CS 247 - Scientific Visualization, KAUST
// Programming Assignment #5 - Complete Solution

#include <cstring>
#include <array>
#include "CS247_prog.h"

// =============================================
// Global variables for this assignment
// =============================================

// Colormap
static int   colormapMode = 0;   // 0=off, 1=rainbow, 2=cool-warm
static float blendFactor = 1.0f;

// Integration method: 0=Euler, 1=RK2, 2=RK4
static int integrationMethod = 0;

// Arrow length mode: 0=constant, 1=magnitude-dependent
static int arrowLengthMode = 0;

// Stored lines (survive timestep changes)
static std::vector<std::vector<glm::vec2>> streamlines;
static std::vector<std::vector<glm::vec2>> pathlines;
static std::vector<glm::ivec2>             streamlineSeeds;
static std::vector<glm::ivec3>             pathlineSeeds; // x,y,t

// GL objects
static GLuint lineVAO = 0, lineVBO = 0;
static GLuint glyphVAO = 0, glyphVBO = 0;

// Rake mode: 0=off, 1=horizontal, 2=vertical
static int rakeMode = 0;

// =============================================
// Helper utilities
// =============================================

static glm::vec2 getVector(int x, int y, int t) {
    x = std::max(0, std::min(x, (int)vol_dim[0] - 1));
    y = std::max(0, std::min(y, (int)vol_dim[1] - 1));
    t = std::max(0, std::min(t, num_timesteps - 1));
    int idx = 3 * (y * vol_dim[0] + x) + 3 * t * data_size;
    return glm::vec2(vector_array[idx], vector_array[idx + 1]);
}

static glm::vec2 bilinearVector(float fx, float fy, int t) {
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1, y1 = y0 + 1;
    float tx = fx - x0, ty = fy - y0;
    glm::vec2 v00 = getVector(x0, y0, t);
    glm::vec2 v10 = getVector(x1, y0, t);
    glm::vec2 v01 = getVector(x0, y1, t);
    glm::vec2 v11 = getVector(x1, y1, t);
    return glm::mix(glm::mix(v00, v10, tx), glm::mix(v01, v11, tx), ty);
}

static glm::vec2 trilinearVector(float fx, float fy, float ft) {
    int t0 = (int)ft, t1 = t0 + 1;
    float tt = ft - t0;
    glm::vec2 v0 = bilinearVector(fx, fy, std::max(0, std::min(t0, num_timesteps - 1)));
    glm::vec2 v1 = bilinearVector(fx, fy, std::max(0, std::min(t1, num_timesteps - 1)));
    return glm::mix(v0, v1, tt);
}

static glm::vec2 gridToNDC(float gx, float gy) {
    return glm::vec2(2.0f * gx / vol_dim[0] - 1.0f,
        2.0f * gy / vol_dim[1] - 1.0f);
}

static glm::vec2 screenToGrid(double sx, double sy) {
    float gx = (float)sx / view_width * vol_dim[0];
    float gy = (float)(view_height - sy) / view_height * vol_dim[1];
    return glm::vec2(gx, gy);
}

static bool inBounds(float x, float y) {
    return x >= 0 && x < vol_dim[0] - 1 && y >= 0 && y < vol_dim[1] - 1;
}

// =============================================
// Integration step helpers
// =============================================

static glm::vec2 stepEuler(float x, float y, int t, float stepDt) {
    glm::vec2 v = bilinearVector(x, y, t);
    return glm::vec2(x + stepDt * v.x, y + stepDt * v.y);
}

static glm::vec2 stepRK2(float x, float y, int t, float stepDt) {
    glm::vec2 k1 = bilinearVector(x, y, t);
    glm::vec2 mid = glm::vec2(x, y) + 0.5f * stepDt * k1;
    glm::vec2 k2 = bilinearVector(mid.x, mid.y, t);
    return glm::vec2(x + stepDt * k2.x, y + stepDt * k2.y);
}

static glm::vec2 stepRK4(float x, float y, int t, float stepDt) {
    glm::vec2 k1 = bilinearVector(x, y, t);
    glm::vec2 p2 = glm::vec2(x, y) + 0.5f * stepDt * k1;
    glm::vec2 k2 = bilinearVector(p2.x, p2.y, t);
    glm::vec2 p3 = glm::vec2(x, y) + 0.5f * stepDt * k2;
    glm::vec2 k3 = bilinearVector(p3.x, p3.y, t);
    glm::vec2 p4 = glm::vec2(x, y) + stepDt * k3;
    glm::vec2 k4 = bilinearVector(p4.x, p4.y, t);
    glm::vec2 s = (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
    return glm::vec2(x + stepDt * s.x, y + stepDt * s.y);
}

static glm::vec2 integrateStep(float x, float y, int t, float stepDt) {
    switch (integrationMethod) {
    case 1:  return stepRK2(x, y, t, stepDt);
    case 2:  return stepRK4(x, y, t, stepDt);
    default: return stepEuler(x, y, t, stepDt);
    }
}

// =============================================
// Streamline computation
// =============================================

static std::vector<glm::vec2> traceStreamline(float sx, float sy, int t) {
    std::vector<glm::vec2> pts;
    const float maxLen = 300.0f;

    // Forward
    float x = sx, y = sy, accumulated = 0.0f;
    pts.push_back(glm::vec2(x, y));
    while (inBounds(x, y) && accumulated < maxLen) {
        glm::vec2 vec = bilinearVector(x, y, t);
        if (glm::length(vec) < 1e-5f) break;
        glm::vec2 nxt = integrateStep(x, y, t, (float)dt);
        accumulated += glm::length(nxt - glm::vec2(x, y));
        x = nxt.x; y = nxt.y;
        pts.push_back(glm::vec2(x, y));
    }

    // Backward
    x = sx; y = sy; accumulated = 0.0f;
    std::vector<glm::vec2> back;
    while (inBounds(x, y) && accumulated < maxLen) {
        glm::vec2 vec = bilinearVector(x, y, t);
        if (glm::length(vec) < 1e-5f) break;
        glm::vec2 nxt = integrateStep(x, y, t, -(float)dt);
        accumulated += glm::length(nxt - glm::vec2(x, y));
        x = nxt.x; y = nxt.y;
        back.push_back(glm::vec2(x, y));
    }
    std::reverse(back.begin(), back.end());
    pts.insert(pts.begin(), back.begin(), back.end());
    return pts;
}

void computeStreamline(int gx, int gy) {
    streamlineSeeds.push_back(glm::ivec2(gx, gy));
    streamlines.push_back(traceStreamline((float)gx, (float)gy, loaded_timestep));
}

static void recomputeAllStreamlines() {
    streamlines.clear();
    for (auto& seed : streamlineSeeds)
        streamlines.push_back(traceStreamline((float)seed.x, (float)seed.y, loaded_timestep));
}

// =============================================
// Pathline computation
// =============================================

static glm::vec2 integratePathStep(float x, float y, float ft, float stepDt) {
    switch (integrationMethod) {
    case 1: { // RK2
        glm::vec2 k1 = trilinearVector(x, y, ft);
        glm::vec2 mid = glm::vec2(x, y) + 0.5f * stepDt * k1;
        glm::vec2 k2 = trilinearVector(mid.x, mid.y, ft + 0.5f * stepDt);
        return glm::vec2(x + stepDt * k2.x, y + stepDt * k2.y);
    }
    case 2: { // RK4
        glm::vec2 k1 = trilinearVector(x, y, ft);
        glm::vec2 p2 = glm::vec2(x, y) + 0.5f * stepDt * k1;
        glm::vec2 k2 = trilinearVector(p2.x, p2.y, ft + 0.5f * stepDt);
        glm::vec2 p3 = glm::vec2(x, y) + 0.5f * stepDt * k2;
        glm::vec2 k3 = trilinearVector(p3.x, p3.y, ft + 0.5f * stepDt);
        glm::vec2 p4 = glm::vec2(x, y) + stepDt * k3;
        glm::vec2 k4 = trilinearVector(p4.x, p4.y, ft + stepDt);
        glm::vec2 s = (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
        return glm::vec2(x + stepDt * s.x, y + stepDt * s.y);
    }
    default: { // Euler
        glm::vec2 v = trilinearVector(x, y, ft);
        return glm::vec2(x + stepDt * v.x, y + stepDt * v.y);
    }
    }
}

static std::vector<glm::vec2> tracePathline(float sx, float sy, int t0) {
    std::vector<glm::vec2> pts;
    const float maxLen = 300.0f;
    float ft = (float)t0;

    // Forward
    float x = sx, y = sy, accumulated = 0.0f;
    pts.push_back(glm::vec2(x, y));
    while (inBounds(x, y) && ft < num_timesteps - 1 && accumulated < maxLen) {
        glm::vec2 vec = trilinearVector(x, y, ft);
        if (glm::length(vec) < 1e-5f) break;
        glm::vec2 nxt = integratePathStep(x, y, ft, (float)dt);
        ft += (float)dt;
        accumulated += glm::length(nxt - glm::vec2(x, y));
        x = nxt.x; y = nxt.y;
        pts.push_back(glm::vec2(x, y));
    }

    // Backward
    x = sx; y = sy; ft = (float)t0; accumulated = 0.0f;
    std::vector<glm::vec2> back;
    while (inBounds(x, y) && ft > 0 && accumulated < maxLen) {
        glm::vec2 vec = trilinearVector(x, y, ft);
        if (glm::length(vec) < 1e-5f) break;
        glm::vec2 nxt = integratePathStep(x, y, ft, -(float)dt);
        ft -= (float)dt;
        accumulated += glm::length(nxt - glm::vec2(x, y));
        x = nxt.x; y = nxt.y;
        back.push_back(glm::vec2(x, y));
    }
    std::reverse(back.begin(), back.end());
    pts.insert(pts.begin(), back.begin(), back.end());
    return pts;
}

void computePathline(int gx, int gy, int t) {
    pathlineSeeds.push_back(glm::ivec3(gx, gy, t));
    pathlines.push_back(tracePathline((float)gx, (float)gy, t));
}

// =============================================
// Drawing lines (streamlines / pathlines)
// =============================================

static void drawLines(const std::vector<std::vector<glm::vec2>>& lines, glm::vec4 color) {
    if (lines.empty()) return;

    std::vector<float> verts;
    for (auto& line : lines) {
        for (auto& p : line) {
            glm::vec2 ndc = gridToNDC(p.x, p.y);
            verts.push_back(ndc.x);
            verts.push_back(ndc.y);
            verts.push_back(0.0f);
        }
    }
    if (verts.empty()) return;

    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    vectorProgram.setUniform("colormapMode", 0);
    vectorProgram.setUniform("vertexColor", color);
    vectorProgram.setUniform("model", glm::mat4(1));

    int offset = 0;
    for (auto& line : lines) {
        if (line.size() >= 2)
            glDrawArrays(GL_LINE_STRIP, offset, (GLsizei)line.size());
        offset += (int)line.size();
    }
    glBindVertexArray(0);
}

// =============================================
// Glyph drawing  (FIX 2: no std::array lambda)
// =============================================

void drawGlyphs() {
    std::vector<float> verts;

    int step = std::max(1, (int)vol_dim[0] / sampling_rate);

    float maxMag = 1e-6f;
    if (arrowLengthMode == 1) {
        for (int gy = 0; gy < vol_dim[1]; gy++)
            for (int gx = 0; gx < vol_dim[0]; gx++)
                maxMag = std::max(maxMag, glm::length(bilinearVector((float)gx, (float)gy, loaded_timestep)));
    }

    float arrowLen = 1.5f * step;
    float headLen = 0.4f * arrowLen;
    float headWid = 0.3f * arrowLen;

    // Helper: push one NDC point (no lambda, no std::array)
    auto push = [&](glm::vec2 p) {
        glm::vec2 n = gridToNDC(p.x, p.y);
        verts.push_back(n.x);
        verts.push_back(n.y);
        verts.push_back(0.0f);
        };

    for (int gy = step / 2; gy < vol_dim[1]; gy += step) {
        for (int gx = step / 2; gx < vol_dim[0]; gx += step) {
            glm::vec2 v = bilinearVector((float)gx, (float)gy, loaded_timestep);
            float     mag = glm::length(v);
            if (mag < 1e-10f) continue;

            glm::vec2 dir = v / mag;
            float     len = (arrowLengthMode == 1) ? arrowLen * (mag / maxMag) : arrowLen;

            glm::vec2 tail = glm::vec2(gx, gy) - dir * (len * 0.5f);
            glm::vec2 head = glm::vec2(gx, gy) + dir * (len * 0.5f);
            glm::vec2 perp = glm::vec2(-dir.y, dir.x);
            glm::vec2 hBase = head - dir * headLen;
            glm::vec2 hL = hBase + perp * (headWid * 0.5f);
            glm::vec2 hR = hBase - perp * (headWid * 0.5f);

            push(tail); push(head);   // shaft
            push(head); push(hL);     // arrowhead left edge
            push(head); push(hR);     // arrowhead right edge
            push(hL);   push(hR);     // arrowhead base
        }
    }

    if (verts.empty()) return;

    glBindVertexArray(glyphVAO);
    glBindBuffer(GL_ARRAY_BUFFER, glyphVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    vectorProgram.setUniform("colormapMode", 0);
    vectorProgram.setUniform("vertexColor", glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
    vectorProgram.setUniform("model", glm::mat4(1));
    glDrawArrays(GL_LINES, 0, (GLsizei)(verts.size() / 3));
    glBindVertexArray(0);
}

// =============================================
// Rake seeding (bonus)
// =============================================

static void seedRake() {
    const int n = 10;
    if (rakeMode == 1) {
        int y = vol_dim[1] / 2;
        for (int i = 0; i < n; i++) {
            int x = (i + 1) * vol_dim[0] / (n + 1);
            computeStreamline(x, y);
            computePathline(x, y, loaded_timestep);
        }
    }
    else if (rakeMode == 2) {
        int x = vol_dim[0] / 2;
        for (int i = 0; i < n; i++) {
            int y = (i + 1) * vol_dim[1] / (n + 1);
            computeStreamline(x, y);
            computePathline(x, y, loaded_timestep);
        }
    }
}

// =============================================
// Background color cycling
// =============================================

static void nextClearColor() {
    clearColor = (++clearColor) % 3;
    switch (clearColor) {
    case 0:  glClearColor(0.0f, 0.0f, 0.0f, 1.0f); break;
    case 1:  glClearColor(0.2f, 0.2f, 0.3f, 1.0f); break;
    default: glClearColor(0.7f, 0.7f, 0.7f, 1.0f); break;
    }
}

// =============================================
// GLFW Callbacks
// =============================================

void frameBufferCallback(GLFWwindow* window, int width, int height) {
    view_width = width; view_height = height;
    glViewport(0, 0, width, height);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_RELEASE) return;
    switch (key) {
    case '1':
        toggle_xy = 0; LoadData(filenames[0]); loaded_file = 0;
        fprintf(stderr, "Loading %s\n", filenames[0]); break;
    case '2':
        toggle_xy = 0; LoadData(filenames[1]); loaded_file = 1;
        fprintf(stderr, "Loading %s\n", filenames[1]); break;
    case '3':
        toggle_xy = 1; LoadData(filenames[2]); loaded_file = 2;
        fprintf(stderr, "Loading %s\n", filenames[2]); break;
    case '0':
        if (num_timesteps > 1) {
            loadNextTimestep();
            recomputeAllStreamlines();
            fprintf(stderr, "Timestep %d\n", loaded_timestep);
        }
        break;
    case GLFW_KEY_A:
        en_arrow = !en_arrow;
        fprintf(stderr, "%s arrows\n", en_arrow ? "Enabling" : "Disabling"); break;
    case GLFW_KEY_L:
        arrowLengthMode = (arrowLengthMode + 1) % 2;
        fprintf(stderr, "Arrow length: %s\n", arrowLengthMode ? "magnitude" : "constant"); break;
    case GLFW_KEY_S:
        current_scalar_field = (current_scalar_field + 1) % num_scalar_fields;
        DownloadScalarFieldAsTexture();
        fprintf(stderr, "Scalar field %d\n", current_scalar_field); break;
    case GLFW_KEY_B:
        nextClearColor(); break;
    case GLFW_KEY_EQUAL:
        sampling_rate = std::min(sampling_rate + 5, 100);
        fprintf(stderr, "Sampling rate: %d\n", sampling_rate); break;
    case GLFW_KEY_MINUS:
        sampling_rate = std::max(sampling_rate - 5, 5);
        fprintf(stderr, "Sampling rate: %d\n", sampling_rate); break;
    case GLFW_KEY_I:
        dt = std::min(dt + 0.005, 1.0);
        fprintf(stderr, "dt: %.4f\n", dt); break;
    case GLFW_KEY_K:
        dt = std::max(dt - 0.005, 0.0001);
        fprintf(stderr, "dt: %.4f\n", dt); break;
    case GLFW_KEY_T:
        en_streamline = !en_streamline;
        fprintf(stderr, "%s streamlines\n", en_streamline ? "Enabling" : "Disabling"); break;
    case GLFW_KEY_P:
        en_pathline = !en_pathline;
        fprintf(stderr, "%s pathlines\n", en_pathline ? "Enabling" : "Disabling"); break;
    case GLFW_KEY_C:
        colormapMode = (colormapMode + 1) % 3;
        fprintf(stderr, "Colormap: %s\n",
            colormapMode == 0 ? "off" : colormapMode == 1 ? "rainbow" : "cool-warm"); break;
    case GLFW_KEY_V:
        blendFactor = std::min(blendFactor + 0.1f, 1.0f);
        fprintf(stderr, "Blend factor: %.1f\n", blendFactor); break;
    case GLFW_KEY_X:
        blendFactor = std::max(blendFactor - 0.1f, 0.0f);
        fprintf(stderr, "Blend factor: %.1f\n", blendFactor); break;
    case GLFW_KEY_M:
        integrationMethod = (integrationMethod + 1) % 3;
        fprintf(stderr, "Integration: %s\n",
            integrationMethod == 0 ? "Euler" : integrationMethod == 1 ? "RK2" : "RK4"); break;
    case GLFW_KEY_D:
        streamlines.clear(); streamlineSeeds.clear();
        pathlines.clear();   pathlineSeeds.clear();
        fprintf(stderr, "Cleared all seeds\n"); break;
    case GLFW_KEY_R:
        rakeMode = (rakeMode + 1) % 3;
        fprintf(stderr, "Rake: %s\n",
            rakeMode == 0 ? "off" : rakeMode == 1 ? "horizontal" : "vertical");
        if (rakeMode != 0) seedRake(); break;
    case GLFW_KEY_Q:
    case GLFW_KEY_ESCAPE:
        exit(0); break;
    default:
        fprintf(stderr,
            "\n--- Keyboard Controls ---\n"
            "1/2/3     load dataset\n"
            "0         next timestep\n"
            "a         toggle arrows\n"
            "l         toggle arrow length (constant/magnitude)\n"
            "+/-       adjust sampling rate\n"
            "t         toggle streamlines\n"
            "p         toggle pathlines\n"
            "m         cycle integration (Euler/RK2/RK4)\n"
            "i/k       increase/decrease dt\n"
            "c         cycle colormap (off/rainbow/cool-warm)\n"
            "v/x       increase/decrease blend factor\n"
            "s         cycle scalar field\n"
            "r         cycle rake (off/horizontal/vertical)\n"
            "d         clear all seeds\n"
            "b         cycle background color\n"
            "LClick    seed streamline/pathline\n"
            "q/Esc     quit\n");
        break;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        glm::vec2 g = screenToGrid(xpos, ypos);
        int gx = (int)g.x, gy = (int)g.y;
        if (gx >= 0 && gx < vol_dim[0] && gy >= 0 && gy < vol_dim[1]) {
            if (en_streamline) computeStreamline(gx, gy);
            if (en_pathline)   computePathline(gx, gy, loaded_timestep);
        }
    }
}

static void errorCallback(int error, const char* description) {
    fprintf(stderr, "GLFW Error: %s\n", description);
}

// =============================================
// Data loading
// =============================================

void loadNextTimestep(void) {
    loaded_timestep = (loaded_timestep + 1) % num_timesteps;
    DownloadScalarFieldAsTexture();
}

void LoadData(char* base_filename) {
    reset_rendering_props();

    char filename[80];
    strcpy(filename, base_filename);
    strcat(filename, ".gri");

    fprintf(stderr, "Loading grid: %s\n", filename);
    FILE* fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return; }

    char header[40];
    fread(header, sizeof(char), 40, fp);
    fclose(fp);

    // FIX 3: read into int temporaries to avoid unsigned short / %d mismatch
    int dx = 0, dy = 0, dz = 0;
    sscanf(header, "SN4DB %d %d %d %d %d %f",
        &dx, &dy, &dz, &num_scalar_fields, &num_timesteps, &timestep);
    vol_dim[0] = (unsigned short)dx;
    vol_dim[1] = (unsigned short)dy;
    vol_dim[2] = (unsigned short)dz;

    fprintf(stderr, "dim: %d x %d x %d  fields: %d  timesteps: %d\n",
        vol_dim[0], vol_dim[1], vol_dim[2], num_scalar_fields, num_timesteps);

    loaded_timestep = 0;
    LoadVectorData(base_filename);
    glfwSetWindowSize(window, vol_dim[0], vol_dim[1]);
    grid_data_loaded = true;
}

void LoadVectorData(const char* filename) {
    data_size = vol_dim[0] * vol_dim[1] * vol_dim[2];

    delete[] vector_array;
    delete[] scalar_fields;
    delete[] scalar_bounds;
    vector_array = new float[data_size * 3 * num_timesteps];
    scalar_fields = new float[data_size * num_scalar_fields * num_timesteps];
    scalar_bounds = new float[2 * num_scalar_fields * num_timesteps];

    int    num_total_fields = num_scalar_fields + 3;
    float* tmp = new float[data_size * num_total_fields];   // one timestep at a time

    for (int k = 0; k < num_timesteps; k++) {
        char ts_name[80];
        if (num_timesteps > 1)
            sprintf(ts_name, "%s.%.5d.dat", filename, k);
        else
            sprintf(ts_name, "%s.dat", filename);

        FILE* fp = fopen(ts_name, "rb");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", ts_name); continue; }
        fread(tmp, sizeof(float), data_size * num_total_fields, fp);
        fclose(fp);

        for (int i = 0; i < num_scalar_fields; i++) {
            float min_val = 1e9f, max_val = -1e9f;
            int   offset = i * data_size * num_timesteps;

            for (int j = 0; j < data_size; j++) {
                float val = tmp[j * num_total_fields + 3 + i];
                scalar_fields[j + k * data_size + offset] = val;

                if (i == 0) {
                    if (toggle_xy) {
                        vector_array[3 * j + 0 + 3 * k * data_size] = tmp[j * num_total_fields + 1];
                        vector_array[3 * j + 1 + 3 * k * data_size] = tmp[j * num_total_fields + 0];
                        vector_array[3 * j + 2 + 3 * k * data_size] = tmp[j * num_total_fields + 2];
                    }
                    else {
                        vector_array[3 * j + 0 + 3 * k * data_size] = tmp[j * num_total_fields + 0];
                        vector_array[3 * j + 1 + 3 * k * data_size] = tmp[j * num_total_fields + 1];
                        vector_array[3 * j + 2 + 3 * k * data_size] = tmp[j * num_total_fields + 2];
                    }
                }
                min_val = std::min(val, min_val);
                max_val = std::max(val, max_val);
            }
            scalar_bounds[2 * i + k * num_scalar_fields * 2] = min_val;
            scalar_bounds[2 * i + 1 + k * num_scalar_fields * 2] = max_val;
        }

        // Normalize scalar fields to [0,1]
        for (int i = 0; i < num_scalar_fields; i++) {
            int   offset = i * data_size * num_timesteps;
            float lo = scalar_bounds[2 * i + k * num_scalar_fields * 2];
            float hi = scalar_bounds[2 * i + 1 + k * num_scalar_fields * 2];

            if (lo < 0.0f && hi > 0.0f) {
                float scale = 0.5f / std::max(-lo, hi);
                for (int j = 0; j < data_size; j++)
                    scalar_fields[offset + j + k * data_size] =
                    0.5f + scalar_fields[offset + j + k * data_size] * scale;
            }
            else {
                float sign = (hi <= 0.0f) ? -1.0f : 1.0f;
                float scale = (hi - lo > 1e-10f) ? 1.0f / (hi - lo) * sign : 1.0f;
                for (int j = 0; j < data_size; j++)
                    scalar_fields[offset + j + k * data_size] =
                    (scalar_fields[offset + j + k * data_size] - lo) * scale;
            }
        }
    }
    delete[] tmp;
    DownloadScalarFieldAsTexture();
    scalar_data_loaded = true;
}

void DownloadScalarFieldAsTexture(void) {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &scalar_field_texture);
    glBindTexture(GL_TEXTURE_2D, scalar_field_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    int ds = vol_dim[0] * vol_dim[1];
    glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY16, vol_dim[0], vol_dim[1], 0,
        GL_LUMINANCE, GL_FLOAT,
        &scalar_fields[(loaded_timestep + current_scalar_field * num_timesteps) * ds]);
    glDisable(GL_TEXTURE_2D);
}

bool initApplication(int argc, char** argv) {
    std::string version((const char*)glGetString(GL_VERSION));
    std::stringstream stream(version);
    unsigned major, minor; char dot;
    stream >> major >> dot >> minor;
    assert(dot == '.');
    std::cout << "OpenGL Version " << major << "." << minor << std::endl;
    return (major > 2 || (major == 2 && minor >= 0));
}

void reset_rendering_props(void) {
    num_scalar_fields = 0;
    streamlines.clear();     streamlineSeeds.clear();
    pathlines.clear();       pathlineSeeds.clear();
}

// =============================================
// Setup & Render
// =============================================

void setup() {
    LoadData(filenames[0]); loaded_file = 0;
    DownloadScalarFieldAsTexture();

    vectorProgram.compileShader("../../../shaders/vertex.vs");
    vectorProgram.compileShader("../../../shaders/fragment.fs");
    vectorProgram.link();

    quad.init();

    glGenVertexArrays(1, &lineVAO);  glGenBuffers(1, &lineVBO);
    glGenVertexArrays(1, &glyphVAO); glGenBuffers(1, &glyphVBO);
}

void render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Background scalar field with colormap
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, scalar_field_texture);
    vectorProgram.use();
    vectorProgram.setUniform("model", glm::mat4(1));
    vectorProgram.setUniform("vertexColor", glm::vec4(0));
    vectorProgram.setUniform("colormapMode", colormapMode);
    vectorProgram.setUniform("blendFactor", blendFactor);
    quad.render();
    glDisable(GL_TEXTURE_2D);

    // Overlays use solid color (disable colormap)
    vectorProgram.setUniform("colormapMode", 0);

    if (en_arrow)      drawGlyphs();
    if (en_streamline) drawLines(streamlines, glm::vec4(0.0f, 1.0f, 0.5f, 1.0f));
    if (en_pathline)   drawLines(pathlines, glm::vec4(1.0f, 0.5f, 0.0f, 1.0f));
}

// =============================================
// Entry point
// =============================================

int main(int argc, char** argv) {
    view_width = view_height = 0;
    toggle_xy = 0;
    en_arrow = en_streamline = en_pathline = false;
    sampling_rate = 15;
    dt = 0.1;
    reset_rendering_props();
    vector_array = scalar_fields = scalar_bounds = NULL;
    grid_data_loaded = scalar_data_loaded = false;
    current_scalar_field = 0;
    clearColor = 0;

    filenames[0] = (char*)"../../../data/block/c_block";
    filenames[1] = (char*)"../../../data/tube/tube";
    filenames[2] = (char*)"../../../data/hurricane/hurricane_p_tc";

    glfwSetErrorCallback(errorCallback);
    if (!glfwInit()) exit(EXIT_FAILURE);

    window = glfwCreateWindow(gWindowWidth, gWindowHeight,
        "AMCS/CS247 Scientific Visualization", nullptr, nullptr);
    if (!window) { glfwTerminate(); exit(EXIT_FAILURE); }

    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetFramebufferSizeCallback(window, frameBufferCallback);
    glfwMakeContextCurrent(window);

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    gladLoadGL();

    if (!initApplication(argc, argv)) { glfwTerminate(); exit(EXIT_FAILURE); }

    setup();
    keyCallback(window, GLFW_KEY_BACKSLASH, 0, GLFW_PRESS, 0);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return EXIT_SUCCESS;
}
