/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// OpenGL Graphics includes
#include <helper_gl.h>

#if defined(__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <thread>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

// CUDA helper functions
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <helper_cuda.h>

#include "defines.h"
#include "fluidsGL_kernels.cuh"

#define MAX_EPSILON_ERROR 1.0f

const char *sSDKname = "fluidsGL";
// CUDA example code that implements the frequency space version of
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the
// CUDA FFT library (CUFFT) to perform velocity diffusion and to
// force non-divergence in the velocity field at each time step. It uses
// CUDA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step.

void cleanup(void);
void reshape(int x, int y);

// CUFFT plan handle
cufftHandle planr2c;
cufftHandle planc2r;
static cData *vxfield = NULL;
static cData *vyfield = NULL;

cData *hvfield = NULL;
cData *dvfield = NULL;
static int wWidth  = MAX(512, DIM);
static int wHeight = MAX(512, DIM);

static int clicked  = 0;
static int fpsCount = 0;
static int fpsLimit = 1;

static int tilex = 64; // Tile width
static int tiley = 64; // Tile height
static int tidsx = 64; // Tids in X
static int tidsy = 4; // Tids in Y

StopWatchInterface *timer = NULL;

// Particle data
GLuint vbo = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
static cData *particles = NULL; // particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit

char *ref_file         = NULL;
bool g_bQAAddTestForce = true;
int  g_iFrameToCompare = 100;
int  g_TotalErrors     = 0;

bool g_bExitESC = false;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

extern "C" void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
extern "C" void advectVelocity(int tidsx, int tidsy, int tilex, int tiley, cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
extern "C" void diffuseProject(int tidsx, int tidsy, int tilex, int tiley, cData *vx, cData *vy, int dx, int dy, float dt, float visc);
extern "C" void updateVelocity(int tidsx, int tidsy, int tilex, int tiley, cData *v, float *vx, float *vy, int dx, int pdx, int dy);
extern "C" void advectParticles(int tidsx, int tidsy, int tilex, int tiley, GLuint vbo, cData *v, int dx, int dy, float dt);


void simulateFluids(void)
{
    // simulate fluid
    advectVelocity(tidsx, tidsy, tilex, tiley, dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM, DT);
    diffuseProject(tidsx, tidsy, tilex, tiley, vxfield, vyfield, CPADW, DIM, DT, VIS);
    updateVelocity(tidsx, tidsy, tilex, tiley, dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM);
    advectParticles(tidsx, tidsy, tilex, tiley, vbo, dvfield, DIM, DIM, DT);
}

void display(void)
{

    if (!ref_file)
    {
        sdkStartTimer(&timer);
        simulateFluids();
    }

    // render points from vertex buffer
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.41f, 0.41f, 0.98f);
    glPointSize(1);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 0, NULL);
    glDrawArrays(GL_POINTS, 0, DS);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);

    if (ref_file)
    {
        return;
    }

    // Finish timing before swap buffers to avoid refresh sync
    sdkStopTimer(&timer);
    glutSwapBuffers();

    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", DIM, DIM, ifps);
        glutSetWindowTitle(fps);
        fpsCount = 0;
        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }

    glutPostRedisplay();
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void)
{
    static int seed = 72191;
    char sq[22];

    if (ref_file)
    {
        seed *= seed;
        sprintf(sq, "%010d", seed);
        // pull the middle 5 digits out of sq
        sq[8] = 0;
        seed = atoi(&sq[3]);

        return seed/99999.f;
    }
    else
    {
        return rand()/(float)RAND_MAX;
    }
}

void initParticles(cData *p, int dx, int dy, int mode)
{
    int i, j;
    switch (mode)
    {
        case 1:
            for (i = dy/4; i < 3*dy/4; i++)
            {
                for (j = dx/4; j < 3*dx/4; j++)
                {
                    p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
                    p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
                }
            }
            break;

        case 2:
            for (i = 0; i < dy; i++)
            {
                for (j = 0; j < dx; j++)
                {
                    p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
                    p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
                }
            }
            break;

        case 3:
            for (i = dy/8; i < 3*dy/8; i++)
            {
                for (j = 0; j < dx; j++)
                {
                    p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
                    p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
                }
            }

            for (i = 5*dy/8; i < 7*dy/8; i++)
            {
                for (j = 0; j < dx; j++)
                {
                    p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
                    p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
                }
            }           
            break;
    }
}

void click(int button, int updown, int x, int y)
{
    lastx = x;
    lasty = y;
    clicked = !clicked;
}

void motion(int x, int y)
{
    // Convert motion coordinates to domain
    float fx = (lastx / (float)wWidth);
    float fy = (lasty / (float)wHeight);
    int nx = (int)(fx * DIM);
    int ny = (int)(fy * DIM);

    if (clicked && nx < DIM-FR && nx > FR-1 && ny < DIM-FR && ny > FR-1)
    {
        int ddx = x - lastx;
        int ddy = y - lasty;
        fx = ddx / (float)wWidth;
        fy = ddy / (float)wHeight;
        int spy = ny-FR;
        int spx = nx-FR;
        addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
        lastx = x;
        lasty = y;
    }

    glutPostRedisplay();
}

void reshape(int x, int y)
{
    wWidth = x;
    wHeight = y;
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 1, 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

void simulateForce(int dir)
{
    int i;

    switch (dir)
    {
        case 1:
            lastx = 500;
            lasty = 256;
            for (i = 500; i > 450; i-=2) 
            {
                clicked = 1;
                motion(i,lasty);
            }
            break;

        case 2:
            lastx = 12;
            lasty = 256;

            for (i = 12; i < 72; i+=2) 
            {
                clicked = 1;
                motion(i,lasty);
            }          
            break;

        case 3:
            lastx = 256;
            lasty = 12;

            for (i = 12; i < 72; i+=2) 
            {
                clicked = 1;
                motion(lastx,i);
            }
            break;

        case 4:
            lastx = 256;
            lasty = 500;

            for (i = 500; i > 450; i-=2) 
            {
                clicked = 1;
                motion(lastx,i);
            }        
            break;
    }

    clicked = 0;
}

float timeTest()
{
    int i;
    float timeCount = 0;
    int simCount = 1000;

    for (i = 0; i < simCount; i++)
    {
        std::clock_t start;
        double duration;

        start = std::clock();
        simulateFluids();
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        timeCount += duration;
    }

    timeCount = timeCount / simCount;
    std::cout<<"Calculated time: "<< timeCount << '\n';
    return timeCount;
}

void reset(int mode)
{
    memset(hvfield, 0, sizeof(cData) * DS);
    cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS,
               cudaMemcpyHostToDevice);

    free(particles);
    particles = (cData *)malloc(sizeof(cData) * DS);
    memset(particles, 0, sizeof(cData) * DS);

    initParticles(particles, DIM, DIM, mode);

    cudaGraphicsUnregisterResource(cuda_vbo_resource);

    getLastCudaError("cudaGraphicsUnregisterBuffer failed");

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * DS,
                    particles, GL_DYNAMIC_DRAW_ARB);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

    getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            g_bExitESC = true;
            #if defined (__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
            break;

        case 'r':
            reset(1);
            break;

        case 't':
            reset(2);
            break;

        case 'z':
            reset(3);
            break;                        

        case 'a':
            simulateForce(1);
            break;

        case 'd':
            simulateForce(2);
            break;

        case 'w':
            simulateForce(4);
            break;

        case 's':
            simulateForce(3);
            break;            

        case 'q':
            std::cout<<"Used configuration: tidsx: "<< tidsx << " tidsy: " << tidsy << '\n';
            simulateForce(1);
            timeTest();
            break;

        case 'f':
            {
                using namespace std::this_thread; // sleep_for, sleep_until
                using namespace std::chrono; // nanoseconds, system_clock, seconds

                std::cout<<"Used configuration: tidsx: "<< tidsx << " tidsy: " << tidsy << '\n';
                simulateForce(1);
                timeTest();
                reset(1);
                break;
            }

        case 'g':
            {
                using namespace std::this_thread; // sleep_for, sleep_until
                using namespace std::chrono; // nanoseconds, system_clock, seconds

                int squareSize = 512;

                for (int j = 0; j < log2(squareSize); j++)
                {
                    squareSize /= 2;

                    tilex = squareSize;
                    tiley = squareSize;
                    tidsx = squareSize;
                    tidsy = 2*std::min(1024/tidsx, tiley);
                    int size = log2(tidsy);
                    float timeArr[size];
                    int i;

                    for (i = 0; i < size; i++) 
                    {
                        tidsy /= 2; 

                        std::cout<<"Used configuration: tidsx: "<< tidsx << " tidsy: " << tidsy << '\n';
                        simulateForce(1);
                        timeArr[i] = timeTest();
                        reset(1);
                    }

                    std::ofstream fout("results.csv", std::ios_base::app);
                    fout << "Config: " << squareSize << std::endl;
                    for (auto& item : timeArr) {
                        fout << item*pow(10,6) <<',';
                    }
                    fout << std::endl;
                }
                break;
            }
            
        default:
            break;
    }
}

void cleanup(void)
{
    cudaGraphicsUnregisterResource(cuda_vbo_resource);

    deleteTexture();

    // Free all host and device resources
    free(hvfield);
    free(particles);
    cudaFree(dvfield);
    cudaFree(vxfield);
    cudaFree(vyfield);
    cufftDestroy(planr2c);
    cufftDestroy(planc2r);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);

    sdkDeleteTimer(&timer);
}

int initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("Compute Stable Fluids");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(click);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);


    if (!isGLVersionSupported(1, 5))
    {
        fprintf(stderr, "ERROR: Support for OpenGL 1.5 is missing");
        fflush(stderr);
        return false;
    }

    if (! areGLExtensionsSupported(
            "GL_ARB_vertex_buffer_object"
        ))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    return true;
}

int main(int argc, char **argv)
{
    int devID;
    cudaDeviceProp deviceProps;

#if defined(__linux__)
    char *Xstatus = getenv("DISPLAY");
    if (Xstatus == NULL)
    {
        printf("Waiving execution as X server is not running\n");
        exit(EXIT_WAIVED);
    }
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s Starting...\n\n", sSDKname);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        exit(EXIT_SUCCESS);
    }

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaDevice(argc, (const char **)argv);

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n",
           deviceProps.name, deviceProps.multiProcessorCount);

    // automated build testing harness
    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
    }

    // Allocate and initialize host data
    GLint bsize;

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    hvfield = (cData *)malloc(sizeof(cData) * DS);
    memset(hvfield, 0, sizeof(cData) * DS);

    // Allocate and initialize device data
    cudaMallocPitch((void **)&dvfield, &tPitch, sizeof(cData)*DIM, DIM);

    cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS,
               cudaMemcpyHostToDevice);
    // Temporary complex velocity field data
    cudaMalloc((void **)&vxfield, sizeof(cData) * PDS);
    cudaMalloc((void **)&vyfield, sizeof(cData) * PDS);

    setupTexture(DIM, DIM);

    // Create particle array
    particles = (cData *)malloc(sizeof(cData) * DS);
    memset(particles, 0, sizeof(cData) * DS);

    initParticles(particles, DIM, DIM, 1);

    // Create CUFFT transform plan configuration
    checkCudaErrors(cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R));

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * DS,
                    particles, GL_DYNAMIC_DRAW_ARB);

    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

    if (bsize != (sizeof(cData) * DS))
        goto EXTERR;

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
    getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

    if (ref_file) {}
    else
    {
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif
        glutMainLoop();
    }

    if (!ref_file)
    {
        exit(EXIT_SUCCESS);
    }

    return 0;

EXTERR:
    printf("Failed to initialize GL extensions.\n");

    exit(EXIT_FAILURE);
}
