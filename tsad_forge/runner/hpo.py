"""하이퍼파라미터 탐색 프로토콜 (리뷰 P1 — "튜닝 예산이라는 숨은 변수").

원칙 (TSB-AD의 Tuning/Eval 분리 관례를 따른다):
1. **평가 entity의 라벨로 고르지 않는다.** 튜닝은 평가 프로파일과 겹치지 않는 별도
   tuning entity에서만 한다. 겹치면 실행을 거부한다.
2. **모든 모델에 같은 예산.** 모델당 n_trials번의 시드 고정 랜덤 탐색. trial 0은 항상
   기본값이다 → 튜닝 결과는 tuning entity에서 기본값보다 나쁠 수 없다.
3. **기본값 결과와 섞지 않는다.** 선택된 파라미터는 `selected` 파일로 저장되고, 평가
   프로파일이 `tuned_params:`로 참조하면 `<model>@tuned` 이름으로 별도 행이 된다.
   리더보드에 default와 tuned가 나란히 보여 "튜닝 이득"이 드러난다.

스터디 파일 (configs/hpo/*.yaml):
    name: lite
    eval_profile: configs/lite.yaml     # defaults·모델 기본 params 상속 + 겹침 검사 대상
    spaces: configs/hpo/spaces.yaml     # 모델별 탐색 공간
    models: [sub_pca, iforest, ...]     # 탐색할 모델
    n_trials: 12                        # 모델당 예산 (trial 0 = 기본값 포함)
    seed: 0
    metric: vus_pr
    results_dir: benchmarks/hpo/lite    # trial 결과 (감사용, 리더보드와 분리)
    output: configs/hpo/selected/lite.yaml
    tuning_datasets: [...]              # 평가와 겹치지 않는 entity

탐색 공간 문법 (파라미터별):
    [a, b, c]                 # 이산 선택
    {log_uniform: [lo, hi]}   # 로그 균등 (실수)
    {uniform: [lo, hi]}       # 균등 (실수)
    {int: [lo, hi]}           # 정수 균등 (양끝 포함)
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import yaml

TRIAL_SEP = "#t"
TUNED_SUFFIX = "@tuned"


# --- 탐색 공간 ---------------------------------------------------------------


def _sample_param(spec, rng: np.random.Generator):
    if isinstance(spec, list):
        return spec[int(rng.integers(len(spec)))]
    if isinstance(spec, dict) and len(spec) == 1:
        ((kind, (lo, hi)),) = spec.items()
        if kind == "log_uniform":
            return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        if kind == "uniform":
            return float(rng.uniform(lo, hi))
        if kind == "int":
            return int(rng.integers(int(lo), int(hi) + 1))
    raise ValueError(f"bad search-space spec: {spec!r}")


def sample_trials(space: dict, n_trials: int, seed: int) -> list[dict]:
    """trial 0 = {} (기본값), 이후 시드 고정 랜덤 샘플. 중복 조합은 다시 뽑는다."""
    rng = np.random.default_rng(seed)
    trials: list[dict] = [{}]
    seen: set = set()
    attempts = 0
    while len(trials) < n_trials and space:
        t = {k: _sample_param(v, rng) for k, v in sorted(space.items())}
        key = tuple(sorted(t.items()))
        attempts += 1
        if key in seen and attempts < 50 * n_trials:
            continue  # 이산 공간이 작으면 중복이 생긴다 — 가능한 한 피한다
        seen.add(key)
        trials.append(t)
    return trials


def accepted_params(model: str) -> set[str]:
    """모델 생성자가 이름으로 받는 인자 집합 (MRO를 따라 **kw로 위임되는 것까지).

    BaseDetector의 **params는 포함하지 않는다 — 거기로 흘러간 오타는 조용히 무시되므로
    탐색 공간 검증에서는 "받지 않는 인자"로 취급해야 한다.
    """
    from tsad_forge.models.base import BaseDetector
    from tsad_forge.models.registry import get_model_class

    names: set[str] = set()
    for cls in get_model_class(model).__mro__:
        if cls is BaseDetector or not issubclass(cls, BaseDetector):
            break
        if "__init__" not in cls.__dict__:
            continue
        sig = inspect.signature(cls.__dict__["__init__"])
        names |= {
            n
            for n, p in sig.parameters.items()
            if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        }
        if not any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            break  # 더 위로 위임하지 않는다
    return names - {"seed", "device"}


def validate_space(model: str, space: dict) -> None:
    allowed = accepted_params(model)
    unknown = sorted(set(space) - allowed)
    if unknown:
        raise ValueError(
            f"{model}: unknown hyperparameters {unknown} (accepted: {sorted(allowed)})"
        )


# --- entity 정체성과 겹침 검사 ----------------------------------------------


def _entity_key(spec: dict) -> tuple:
    params = {k: v for k, v in (spec.get("data_params") or {}).items() if k != "data_dir"}
    return (str(spec["data"]), tuple(sorted((k, str(v)) for k, v in params.items())))


def check_disjoint(tuning: list[dict], evaluation: list[dict]) -> None:
    """tuning entity와 평가 entity가 겹치면 ValueError — 평가 라벨로 고르는 것을 막는다.

    주의: 기본 synthetic(data_params 없음)은 시드마다 다른 데이터를 생성하므로 tuning
    쪽에서는 반드시 data_params.seed를 명시해야 한다 (여기서 강제한다).
    """
    ev = {_entity_key(s) for s in evaluation}
    clash = [s for s in tuning if _entity_key(s) in ev]
    if clash:
        raise ValueError(f"tuning entities overlap the evaluation profile: {clash}")
    for s in tuning:
        if s["data"] == "synthetic" and "seed" not in (s.get("data_params") or {}):
            raise ValueError("synthetic tuning entities must set data_params.seed explicitly")


# --- 스터디 실행 -------------------------------------------------------------


def _data_specs(raw: list) -> list[dict]:
    return [{"data": d} if isinstance(d, str) else dict(d) for d in raw]


def load_study(path: str | Path) -> dict:
    from tsad_forge.runner.matrix import _expand_datasets, load_profile

    study = yaml.safe_load(Path(path).read_text())
    study["_eval"] = load_profile(study["eval_profile"])
    spaces_src = study.get("spaces", {})
    spaces = (
        yaml.safe_load(Path(spaces_src).read_text()) if isinstance(spaces_src, str) else spaces_src
    )
    study["_spaces"] = spaces
    for m in study["models"]:
        if m not in spaces:
            raise ValueError(f"no search space for model '{m}'")
        validate_space(m, spaces[m])
    tuning, _ = _expand_datasets({"datasets": study["tuning_datasets"]})
    evaluation, _ = _expand_datasets(study["_eval"])
    check_disjoint(tuning, evaluation)
    study["_tuning"] = tuning
    return study


def _eval_model_params(eval_profile: dict) -> dict[str, dict]:
    """평가 프로파일에 적힌 모델별 기본 model_params (예: DL 공통 학습 예산)."""
    out: dict[str, dict] = {}
    for spec in eval_profile["models"]:
        spec = {"model": spec} if isinstance(spec, str) else spec
        if not spec.get("model_alias"):
            out.setdefault(spec["model"], dict(spec.get("model_params") or {}))
    return out


def build_trial_jobs(study: dict):
    from tsad_forge.runner.experiment import plan_experiment
    from tsad_forge.runner.matrix import Job

    ev = study["_eval"]
    defaults = {k: v for k, v in ev.get("defaults", {}).items()}
    base_params = _eval_model_params(ev)
    seed = int((ev.get("seeds") or [0])[0])
    results_dir = study["results_dir"]
    trials_by_model: dict[str, list[dict]] = {}
    jobs = []
    for model in study["models"]:
        trials = sample_trials(
            study["_spaces"][model], int(study["n_trials"]), int(study.get("seed", 0))
        )
        trials_by_model[model] = trials
        for k, params in enumerate(trials):
            for data_spec in study["_tuning"]:
                overrides = {
                    **defaults,
                    **data_spec,
                    "model": model,
                    "model_alias": f"{model}{TRIAL_SEP}{k:02d}",
                    "model_params": {**base_params.get(model, {}), **params},
                    "seed": seed,
                    "results_dir": results_dir,
                    "save_raw_scores": False,
                }
                plan = plan_experiment(overrides)
                jobs.append(
                    Job(
                        overrides=overrides,
                        run_id=plan.run_id,
                        lane="torch" if plan.uses_torch else "cpu",
                        tag=(plan.name, plan.dataset, plan.channel, seed),
                        exists=plan.exists(),
                    )
                )
    return jobs, trials_by_model


def select_best(study: dict, trials_by_model: dict[str, list[dict]]) -> dict:
    """trial별 tuning entity 평균 지표 → 모델별 최고 trial. 일부 entity가 실패한 trial은 제외."""
    from tsad_forge.runner.results import add_entity_column, load_all_results

    metric = study.get("metric", "vus_pr")
    df = load_all_results(study["results_dir"])
    df = add_entity_column(df[df["metric"] == metric])
    n_entities = len(study["_tuning"])
    selected: dict = {}
    for model, trials in trials_by_model.items():
        rows = []
        for k, params in enumerate(trials):
            vals = df[df["model"] == f"{model}{TRIAL_SEP}{k:02d}"]["value"]
            if len(vals) == n_entities:
                rows.append((float(vals.mean()), -k, k, params))
        if not rows:
            continue
        default = next((r[0] for r in rows if r[2] == 0), float("nan"))
        best = max(rows)  # 동점이면 낮은 trial 번호 (기본값 우선)
        selected[model] = {
            "params": best[3],
            "trial": best[2],
            f"tuning_{metric}": round(best[0], 6),
            f"default_{metric}": round(default, 6) if not math.isnan(default) else None,
            "n_trials_completed": len(rows),
        }
    return selected


def write_selected(study: dict, selected: dict) -> Path:
    out = Path(study["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "_provenance": {
            "study": study["name"],
            "eval_profile": study["eval_profile"],
            "metric": study.get("metric", "vus_pr"),
            "n_trials_budget": int(study["n_trials"]),
            "search_seed": int(study.get("seed", 0)),
            "tuning_entities": [
                {
                    "data": s["data"],
                    **({"data_params": s["data_params"]} if s.get("data_params") else {}),
                }
                for s in study["_tuning"]
            ],
            "note": "selected on tuning entities disjoint from the evaluation profile",
        },
        **selected,
    }
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return out


def load_selected(path: str | Path) -> dict[str, dict]:
    """selected 파일 → {model: params}."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    return {m: dict(v.get("params") or {}) for m, v in doc.items() if not m.startswith("_")}
