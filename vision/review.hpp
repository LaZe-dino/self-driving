#pragma once

#include <string>

// Steps through a logged session, drawing the stored labels over the stored
// image so the dataset itself can be checked. With `out_dir` set, writes the
// renders instead of opening a window. Human labels go to labels.jsonl.
int review_session(const std::string& session_dir, const std::string& out_dir);
