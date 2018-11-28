//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifdef ANDROID

#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <vector>

#include "CreateGLBuffer.hpp"

#define EGL_RESULT_CHECK(X) do { \
                                   EGLint error = eglGetError(); \
                                   if (!(X) || error != EGL_SUCCESS) { \
                                       std::cerr << \
                                          "EGL error " << error << " at " << __FILE__ << ":" << __LINE__ <<std::endl;\
                                       std::exit(1); \
                                    } \
                            } while (0)

CreateGLBuffer::CreateGLBuffer() {
    this->createGLContext();
}

CreateGLBuffer::~CreateGLBuffer() {
}

void CreateGLBuffer::createGLContext() {
    const EGLint attribListWindow[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_STENCIL_SIZE, 0,
        EGL_NONE
    };
    const EGLint attribListPbuffer[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_STENCIL_SIZE, 0,
        EGL_NONE
    };
    const EGLint srfPbufferAttr[] = {
        EGL_WIDTH, 512,
        EGL_HEIGHT, 512,
        EGL_LARGEST_PBUFFER, EGL_TRUE,
        EGL_NONE
    };
    static const EGLint gl_context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };

    EGLDisplay eglDisplay = 0;
    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGL_RESULT_CHECK(eglDisplay != EGL_NO_DISPLAY);

    EGLint iMajorVersion, iMinorVersion;
    EGL_RESULT_CHECK(eglInitialize(eglDisplay, &iMajorVersion, &iMinorVersion));

    EGLConfig eglConfigWindow = 0;
    int iConfigs = 0;
    EGL_RESULT_CHECK(eglChooseConfig(eglDisplay, attribListWindow, &eglConfigWindow, 1, &iConfigs));

    EGLSurface eglSurfacePbuffer = 0;
    eglSurfacePbuffer = eglCreatePbufferSurface(eglDisplay, eglConfigWindow,srfPbufferAttr);
    EGL_RESULT_CHECK(eglSurfacePbuffer != EGL_NO_SURFACE);

    EGLContext eglContext = 0;
    eglContext = eglCreateContext(eglDisplay, eglConfigWindow, EGL_NO_CONTEXT, gl_context_attribs);
    EGL_RESULT_CHECK(eglContext != EGL_NO_CONTEXT);

    EGL_RESULT_CHECK(eglMakeCurrent(eglDisplay, eglSurfacePbuffer, eglSurfacePbuffer, eglContext));
}

GLuint CreateGLBuffer::convertImage2GLBuffer(const std::vector<std::string>& fileLines, const size_t bufSize)
{
   std::cout << "Processing DNN Input: " << std::endl;
   std::vector<uint8_t> inputVec;
   for(size_t i = 0; i < fileLines.size(); ++i) {
      std::string fileLine(fileLines[i]);
      // treat each line as a space-separated list of input files
      std::vector<std::string> filePaths;
      split(filePaths, fileLine, ' ');
      std::string filePath(filePaths[0]);
      std::cout << "\t" << i + 1 << ") " << filePath << std::endl;
      loadByteDataFileBatched(filePath, inputVec, i);
   }
   GLuint userBuffers;
   glGenBuffers(1, &userBuffers);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, userBuffers);
   glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, inputVec.data(), GL_STREAM_DRAW);

   return userBuffers;
}

void CreateGLBuffer::setGPUPlatformConfig(zdl::DlSystem::PlatformConfig& platformConfig)
{
    void* glcontext = eglGetCurrentContext();
    void* gldisplay = eglGetCurrentDisplay();
    zdl::DlSystem::UserGLConfig userGLConfig;
    userGLConfig.userGLContext = glcontext;
    userGLConfig.userGLDisplay = gldisplay;
    zdl::DlSystem::UserGpuConfig userGpuConfig;
    userGpuConfig.userGLConfig = userGLConfig;
    platformConfig.setUserGpuConfig(userGpuConfig);
}

#endif
