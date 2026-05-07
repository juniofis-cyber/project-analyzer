"""
Módulo de calibração para filmes radiocrômicos EBT3/EBT4.

Implementa modelos de fitting baseados nas referências pesquisadas:
- Dosepy (LuisOlivaresJ) — abordagem com LUT / interpolação
- CHROMO (matteobama) — polinomial / spline
- radiochromic-film (ckswilliams) — power law

Modelos disponíveis:
1. Power Law:       Dose = a * NOD^b
2. Polinomial 3º:   Dose = a + b*NOD + c*NOD² + d*NOD³
3. Rational:        Dose = (a + b*NOD) / (1 + c*NOD)
4. Log-linear:      Dose = exp(a + b * log(NOD))
5. Spline cúbica:   Interpolação spline (sem modelo analítico)

O Net Optical Density (NOD) é usado como variável independente (X)
e Dose (Gy) como variável dependente (Y).
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, interp1d
from dataclasses import dataclass
from typing import Literal, Callable
import json


@dataclass
class CalibrationData:
    """Dados de calibração: pares de (NOD, Dose) conhecidos."""
    nod: np.ndarray  # Net Optical Density
    dose: np.ndarray  # Dose em Gy


@dataclass
class CalibrationModel:
    """Modelo de calibração ajustado."""
    model_type: str
    params: np.ndarray | None
    spline: CubicSpline | None = None
    nod_min: float = 0.0
    nod_max: float = 0.0
    dose_min: float = 0.0
    dose_max: float = 0.0
    rmse: float = 0.0
    r_squared: float = 0.0


def _power_law(nod, a, b):
    """Power law: Dose = a * NOD^b"""
    nod_safe = np.clip(nod, a_min=1e-6, a_max=None)
    return a * (nod_safe ** b)


def _polynomial3(nod, a, b, c, d):
    """Polinômio de 3º grau: Dose = a + b*NOD + c*NOD² + d*NOD³"""
    return a + b * nod + c * (nod ** 2) + d * (nod ** 3)


def _polynomial2(nod, a, b, c):
    """Polinômio de 2º grau: Dose = a + b*NOD + c*NOD²"""
    return a + b * nod + c * (nod ** 2)


def _rational(nod, a, b, c):
    """Rational: Dose = (a + b*NOD) / (1 + c*NOD)"""
    return (a + b * nod) / (1.0 + c * nod)


def _log_linear(nod, a, b):
    """Log-linear: Dose = exp(a + b * ln(NOD)) = e^a * NOD^b"""
    nod_safe = np.clip(nod, a_min=1e-6, a_max=None)
    return np.exp(a + b * np.log(nod_safe))


MODEL_FUNCTIONS: dict[str, Callable] = {
    "power_law": _power_law,
    "polynomial3": _polynomial3,
    "polynomial2": _polynomial2,
    "rational": _rational,
    "log_linear": _log_linear,
}


def fit_calibration(
    data: CalibrationData,
    model_type: Literal["power_law", "polynomial3", "polynomial2", "rational", "log_linear", "spline"] = "power_law",
    initial_guess: np.ndarray | None = None,
    bounds: tuple | None = None,
) -> CalibrationModel:
    """
    Ajusta um modelo de calibração aos dados de (NOD, Dose).

    Args:
        data: Objeto CalibrationData com arrays nod e dose
        model_type: Tipo de modelo matemático
        initial_guess: Palpite inicial para otimização (opcional)
        bounds: Limites para parâmetros (opcional)

    Returns:
        CalibrationModel com parâmetros ajustados e métricas de qualidade
    """
    nod = np.asarray(data.nod, dtype=np.float64)
    dose = np.asarray(data.dose, dtype=np.float64)

    # Ordenar por NOD crescente (importante para spline)
    sort_idx = np.argsort(nod)
    nod = nod[sort_idx]
    dose = dose[sort_idx]

    nod_min, nod_max = float(np.min(nod)), float(np.max(nod))
    dose_min, dose_max = float(np.min(dose)), float(np.max(dose))

    if model_type == "spline":
        # Spline cúbica — sem parâmetros analíticos, interpolação direta
        # Usamos suavização leve se houver muitos pontos
        spline = CubicSpline(nod, dose, bc_type="natural")
        predicted = spline(nod)
        model = CalibrationModel(
            model_type="spline",
            params=None,
            spline=spline,
            nod_min=nod_min,
            nod_max=nod_max,
            dose_min=dose_min,
            dose_max=dose_max,
        )
    else:
        func = MODEL_FUNCTIONS[model_type]
        n_params = {
            "power_law": 2,
            "polynomial3": 4,
            "polynomial2": 3,
            "rational": 3,
            "log_linear": 2,
        }[model_type]

        if initial_guess is None:
            # Palpites padrão baseados em heurísticas para filmes EBT
            if model_type == "power_law":
                initial_guess = np.array([1.0, 1.0])
            elif model_type == "polynomial3":
                initial_guess = np.array([0.0, 1.0, 0.0, 0.0])
            elif model_type == "polynomial2":
                initial_guess = np.array([0.0, 1.0, 0.0])
            elif model_type == "rational":
                initial_guess = np.array([0.0, 1.0, 0.0])
            elif model_type == "log_linear":
                initial_guess = np.array([0.0, 1.0])

        # Ajuste com curve_fit
        popt, pcov = curve_fit(
            func,
            nod,
            dose,
            p0=initial_guess,
            bounds=bounds if bounds else (-np.inf, np.inf),
            maxfev=10000,
        )
        predicted = func(nod, *popt)
        model = CalibrationModel(
            model_type=model_type,
            params=popt,
            nod_min=nod_min,
            nod_max=nod_max,
            dose_min=dose_min,
            dose_max=dose_max,
        )

    # Métricas de qualidade do ajuste
    ss_res = np.sum((dose - predicted) ** 2)
    ss_tot = np.sum((dose - np.mean(dose)) ** 2)
    rmse = np.sqrt(ss_res / len(dose))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    model.rmse = float(rmse)
    model.r_squared = float(r_squared)

    return model


def predict_dose(model: CalibrationModel, nod: np.ndarray | float) -> np.ndarray:
    """
    Prediz a dose a partir de valores NOD usando o modelo calibrado.

    Args:
        model: Modelo calibrado
        nod: Valor(es) NOD (pode ser escalar ou array)

    Returns:
        np.ndarray: Dose(s) predita(s) em Gy
    """
    nod_arr = np.asarray(nod, dtype=np.float64)
    is_scalar = nod_arr.ndim == 0
    nod_arr = np.atleast_1d(nod_arr)

    if model.model_type == "spline" and model.spline is not None:
        # Para spline, extrapolamos linearmente fora do intervalo
        result = np.empty_like(nod_arr)
        inside = (nod_arr >= model.nod_min) & (nod_arr <= model.nod_max)
        below = nod_arr < model.nod_min
        above = nod_arr > model.nod_max

        if np.any(inside):
            result[inside] = model.spline(nod_arr[inside])

        # Extrapolação simples nos limites
        if np.any(below):
            # Usar derivada no limite inferior para extrapolar
            deriv_min = model.spline.derivative()(model.nod_min)
            result[below] = model.dose_min + deriv_min * (nod_arr[below] - model.nod_min)
        if np.any(above):
            deriv_max = model.spline.derivative()(model.nod_max)
            result[above] = model.dose_max + deriv_max * (nod_arr[above] - model.nod_max)
    else:
        func = MODEL_FUNCTIONS[model.model_type]
        result = func(nod_arr, *model.params)

    result = np.clip(result, a_min=0.0, a_max=None)
    return float(result[0]) if is_scalar else result


def model_to_dict(model: CalibrationModel) -> dict:
    """Serializa o modelo para dicionário (JSON-friendly)."""
    return {
        "model_type": model.model_type,
        "params": model.params.tolist() if model.params is not None else None,
        "nod_min": model.nod_min,
        "nod_max": model.nod_max,
        "dose_min": model.dose_min,
        "dose_max": model.dose_max,
        "rmse": model.rmse,
        "r_squared": model.r_squared,
    }


def model_from_dict(data: dict) -> CalibrationModel:
    """Desserializa modelo a partir de dicionário."""
    params = np.array(data["params"]) if data["params"] is not None else None
    model = CalibrationModel(
        model_type=data["model_type"],
        params=params,
        nod_min=data["nod_min"],
        nod_max=data["nod_max"],
        dose_min=data["dose_min"],
        dose_max=data["dose_max"],
        rmse=data["rmse"],
        r_squared=data["r_squared"],
    )
    if model.model_type == "spline":
        # Spline não pode ser reconstruída puramente do dict sem dados brutos
        model.spline = None
    return model


def save_model(model: CalibrationModel, filepath: str):
    """Salva modelo em arquivo JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(model_to_dict(model), f, indent=2, ensure_ascii=False)


def load_model(filepath: str) -> CalibrationModel:
    """Carrega modelo de arquivo JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model_from_dict(data)
