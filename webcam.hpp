#pragma once

struct VisionCommand {
    double steering = 0.0;
    double target_speed = 3.0;
    bool engaged = false;
};

bool atlas_webcam_available();
bool atlas_webcam_start();
bool atlas_webcam_pump(VisionCommand& command);
void atlas_webcam_stop();
