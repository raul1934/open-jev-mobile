plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.openjev.mobile"
    compileSdk = 36
    ndkVersion = "27.1.12297006"

    defaultConfig {
        applicationId = "com.openjev.mobile"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            // x86_64 only so the app also runs in the Android emulator.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                // llama.cpp is unusably slow unoptimized, so every variant builds it in Release.
                arguments += listOf(
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=ON",
                    "-DGGML_NATIVE=OFF",
                    // One CPU library per instruction-set level; the best one is picked at runtime.
                    "-DGGML_BACKEND_DL=ON",
                    "-DGGML_CPU_ALL_VARIANTS=ON",
                    "-DGGML_LLAMAFILE=OFF",
                    "-DGGML_OPENMP=OFF",
                    "-DLLAMA_OPENSSL=OFF",
                )
                // Optional: -PllamaSrc=/path/to/llama.cpp (checked out at v0.4.1) to build offline.
                (project.findProperty("llamaSrc") as String?)?.let { arguments += "-DLLAMA_SRC_DIR=$it" }
            }
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
            // Windows: pass -PcxxDir=C:/short/path if the default build path gets too long.
            (project.findProperty("cxxDir") as String?)?.let { buildStagingDirectory = file(it) }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Signed with the local debug key so the APK can be sideloaded as is.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    sourceSets {
        getByName("main") {
            // The test prompts shown in the app are the repository's examples/*.json.
            assets.srcDir("../../examples")
        }
    }

    packaging {
        // ggml loads its CPU backends with dlopen() from nativeLibraryDir, so they must be extracted.
        jniLibs.useLegacyPackaging = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)

    testImplementation(libs.junit)
}
