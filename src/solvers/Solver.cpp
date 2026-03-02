#include "Solver.h"
#include "../utilities/FileUtil.h"

wk_ptr<Hair> Solver::GetHair() {
    if (!hair.expired())
        return hair;
    else
        return kami.lock()->GetHair();
};

wk_ptr<Body> Solver::GetBody() {
    if (!body.expired())
        return body;
    else
        return kami.lock()->GetBody();
};

wk_ptr<Motion> Solver::GetMotion() {
    if (!motion.expired())
        return motion;
    else
        return kami.lock()->GetMotion();
}

Kami::KamiSimulatorParams Solver::GetKamiParams() {
    if (params) {
        return *params;
    } else
        return kami.lock()->GetKamiParams();
}
float Solver::GetKamiInternalStepSize() { return kami.lock()->GetInternalStepSize(); };
string Solver::GenerateKamiFilename(string str) { return kami.lock()->GenerateFileName(str); }

Solver::~Solver() {}

void Solver::UploadHair() {}

void Solver::DownloadHair() {}

string Solver::GetInfoString() { return "No info"; }

void Solver::SubstituteHair(wk_ptr<Hair> _hair) { hair = _hair; }

void Solver::SubstituteBody(wk_ptr<Body> _body) { body = _body; }

void Solver::SubstituteMotion(wk_ptr<Motion> _motion) { motion = _motion; }

void Solver::SubstituteKamiimParams(Kami::KamiSimulatorParams _params) { params = make_shared<Kami::KamiSimulatorParams>(_params); }
