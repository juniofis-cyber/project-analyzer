"""
TPS Parser Universal — Leitor de distribuições de dose de qualquer TPS.

Suporta:
  - DICOM RT Dose (.dcm) — via pydicom (opcional, tenta importar)
  - Eclipse .ALL — parser ASCII customizado
  - CSV/TXT genérico — matriz + metadados em cabeçalho
  - PNG/TIFF dose map — imagem + input manual de escala

Retorna objeto DoseDistribution padronizado para uso no gamma analysis.
"""

import numpy as np
import io
from pathlib import Path
from PIL import Image
import pandas as pd


class DoseDistribution:
    """Container padronizado para distribuição de dose 2D."""
    def __init__(self, dose, resolution_mm, origin_mm=(0.0, 0.0), source="unknown", metadata=None):
        self.dose = np.asarray(dose, dtype=np.float64)
        self.resolution_mm = float(resolution_mm)
        self.origin_mm = tuple(origin_mm)
        self.shape = self.dose.shape
        self.max_dose = float(np.max(self.dose))
        self.min_dose = float(np.min(self.dose))
        self.mean_dose = float(np.mean(self.dose))
        self.source = source
        self.metadata = metadata or {}

    def get_axes_mm(self):
        """Retorna arrays de coordenadas físicas em mm para cada eixo."""
        h, w = self.shape
        res = self.resolution_mm
        x0, y0 = self.origin_mm
        x_axis = x0 + np.arange(w) * res
        y_axis = y0 + np.arange(h) * res
        return x_axis, y_axis


def detect_format(filepath_or_bytes, filename=None):
    """Detecta o formato do arquivo a partir da extensão e conteúdo."""
    if filename is None:
        if hasattr(filepath_or_bytes, 'name'):
            filename = filepath_or_bytes.name
        else:
            filename = str(filepath_or_bytes)

    ext = Path(filename).suffix.lower()

    if ext == '.dcm':
        return 'dicom'
    elif ext == '.all':
        return 'eclipse_all'
    elif ext in ('.csv', '.txt'):
        return 'csv'
    elif ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
        return 'image'
    else:
        # Tentar inferir pelo conteúdo
        content = _read_raw(filepath_or_bytes)
        text = content[:200].decode('utf-8', errors='ignore').lower()
        if b'DICM' in content[:132]:
            return 'dicom'
        elif 'dicom' in text:
            return 'dicom'
        elif ext == '.all' or 'patient' in text or 'plan' in text:
            return 'eclipse_all'
        else:
            return 'csv'


def _read_raw(filepath_or_bytes):
    """Lê conteúdo bruto do arquivo."""
    if isinstance(filepath_or_bytes, (str, Path)):
        with open(filepath_or_bytes, 'rb') as f:
            return f.read()
    elif hasattr(filepath_or_bytes, 'read'):
        return filepath_or_bytes.read()
    elif isinstance(filepath_or_bytes, bytes):
        return filepath_or_bytes
    else:
        raise ValueError(f"Tipo não suportado: {type(filepath_or_bytes)}")


def read_dicom(filepath_or_bytes, filename=None):
    """Lê arquivo DICOM RT Dose."""
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom não instalado. Instale com: pip install pydicom")

    content = _read_raw(filepath_or_bytes)
    ds = pydicom.dcmread(io.BytesIO(content))

    # DICOM RT Dose armazena dose como inteiros × DoseGridScaling
    dose_grid = ds.pixel_array.astype(np.float64)
    if hasattr(ds, 'DoseGridScaling'):
        dose_grid *= float(ds.DoseGridScaling)
    elif hasattr(ds, 'DoseSummationType'):
        pass

    # Metadados espaciais
    resolution_mm = 1.0
    origin_mm = (0.0, 0.0)

    if hasattr(ds, 'ImagePlanePixelSpacing'):
        spacing = ds.ImagePlanePixelSpacing
        resolution_mm = float(spacing[0])
    elif hasattr(ds, 'PixelSpacing'):
        spacing = ds.PixelSpacing
        resolution_mm = float(spacing[0])

    if hasattr(ds, 'ImagePositionPatient'):
        ipp = ds.ImagePositionPatient
        origin_mm = (float(ipp[0]), float(ipp[1]))

    metadata = {
        "modality": getattr(ds, 'Modality', 'RTDOSE'),
        "dose_summation_type": getattr(ds, 'DoseSummationType', ''),
        "dose_units": getattr(ds, 'DoseUnits', 'GY'),
        "dose_type": getattr(ds, 'DoseType', 'PHYSICAL'),
    }

    return DoseDistribution(
        dose=dose_grid,
        resolution_mm=resolution_mm,
        origin_mm=origin_mm,
        source='dicom',
        metadata=metadata,
    )


def read_eclipse_all(filepath_or_bytes, filename=None):
    """Lê arquivo .ALL do Eclipse/Varian (formato ASCII)."""
    content = _read_raw(filepath_or_bytes)

    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = content.decode('latin-1')
        except:
            text = content.decode('utf-8', errors='ignore')

    lines = text.splitlines()

    # Extrair metadados do nome do arquivo
    if filename is None:
        filename = "dose.all"
    basename = Path(filename).stem
    parts = basename.split('.')

    plane = None
    coordinate_mm = None
    patient_id = None
    structure = None

    if len(parts) >= 4:
        patient_id = parts[0]
        structure = parts[1]
        plane = parts[2]
        try:
            coordinate_mm = float(parts[3])
        except ValueError:
            coordinate_mm = None

    # Parser do conteúdo numérico
    data_rows = []
    header_lines = []
    parsing_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        try:
            numeric_tokens = [float(t.replace(',', '.')) for t in tokens]
            data_rows.append(numeric_tokens)
            parsing_data = True
        except ValueError:
            if not parsing_data:
                header_lines.append(stripped)

    # Fallback: split por vírgula
    if len(data_rows) == 0:
        for line in lines:
            stripped = line.strip()
            if not stripped or ',' not in stripped:
                continue
            tokens = stripped.split(',')
            try:
                numeric_tokens = [float(t) for t in tokens]
                data_rows.append(numeric_tokens)
            except ValueError:
                header_lines.append(stripped)

    # Construir matriz
    dose_matrix = None
    if data_rows:
        lengths = [len(r) for r in data_rows]
        if len(set(lengths)) == 1:
            dose_matrix = np.array(data_rows, dtype=np.float64)
        else:
            max_len = max(lengths)
            padded = [r for r in data_rows if len(r) == max_len]
            if padded:
                dose_matrix = np.array(padded, dtype=np.float64)

    if dose_matrix is None:
        raise ValueError("Não foi possível extrair matriz de dose do arquivo .ALL")

    # A dose em .ALL geralmente está em cGy — converter para Gy
    if dose_matrix.max() > 100:
        dose_matrix = dose_matrix / 100.0

    # Inferir resolução
    # FOV típico do Eclipse: ~400 mm, matriz 305×353 ou similar
    h, w = dose_matrix.shape
    fov_mm = 400.0
    resolution_mm = fov_mm / max(h, w)

    # Se tivermos coordenada no nome, usar para calcular origem
    origin_mm = (0.0, 0.0)

    metadata = {
        "filename": filename,
        "patient_id": patient_id,
        "structure": structure,
        "plane": plane,
        "coordinate_mm": coordinate_mm,
        "header_lines": header_lines[:10],
    }

    return DoseDistribution(
        dose=dose_matrix,
        resolution_mm=resolution_mm,
        origin_mm=origin_mm,
        source='eclipse_all',
        metadata=metadata,
    )


def read_csv_dose(filepath_or_bytes, filename=None, delimiter=None, header_rows=0):
    """Lê arquivo CSV/TXT com matriz de dose.

    Formato esperado:
      - Opcional: cabeçalho com metadados (# comentários ou linhas de texto)
      - Matriz de valores numéricos separados por vírgula, espaço ou tab
      - Opcional: linhas com resolution_mm, origin_mm, etc.
    """
    content = _read_raw(filepath_or_bytes)
    text = content.decode('utf-8', errors='ignore')
    lines = text.splitlines()

    # Separar cabeçalho de dados
    header_lines = []
    data_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Linhas que começam com # são comentários
        if stripped.startswith('#'):
            header_lines.append(stripped[1:].strip())
            continue
        # Tentar detectar se é dados ou metadados
        if stripped[0].isdigit() or stripped[0] in '-+.':
            data_lines.append(stripped)
        else:
            header_lines.append(stripped)

    # Tentar detectar delimitador
    if delimiter is None and data_lines:
        first = data_lines[0]
        if ',' in first:
            delimiter = ','
        elif '\t' in first:
            delimiter = '\t'
        else:
            delimiter = None  # espaço

    # Parse dados
    data_rows = []
    for line in data_lines:
        if delimiter:
            tokens = line.split(delimiter)
        else:
            tokens = line.split()
        try:
            numeric_tokens = [float(t.replace(',', '.')) for t in tokens if t.strip()]
            if numeric_tokens:
                data_rows.append(numeric_tokens)
        except ValueError:
            pass

    if not data_rows:
        raise ValueError("Nenhum dado numérico encontrado no CSV/TXT")

    # Construir matriz
    lengths = [len(r) for r in data_rows]
    if len(set(lengths)) == 1:
        dose_matrix = np.array(data_rows, dtype=np.float64)
    else:
        max_len = max(lengths)
        padded = [r for r in data_rows if len(r) == max_len]
        dose_matrix = np.array(padded, dtype=np.float64)

    # Tentar extrair resolução do cabeçalho
    resolution_mm = 1.0
    origin_mm = (0.0, 0.0)

    header_text = ' '.join(header_lines).lower()
    for line in header_lines:
        line_lower = line.lower()
        if 'resolution' in line_lower or 'spacing' in line_lower or 'pixel' in line_lower:
            # Tentar extrair número
            import re
            numbers = re.findall(r'\d+\.?\d*', line)
            if numbers:
                resolution_mm = float(numbers[0])
                if 'cm' in line_lower:
                    resolution_mm *= 10.0
        if 'origin' in line_lower or 'x0' in line_lower or 'y0' in line_lower:
            import re
            numbers = re.findall(r'-?\d+\.?\d*', line)
            if len(numbers) >= 2:
                origin_mm = (float(numbers[0]), float(numbers[1]))

    # Se dose parece estar em cGy
    if dose_matrix.max() > 100:
        dose_matrix = dose_matrix / 100.0

    return DoseDistribution(
        dose=dose_matrix,
        resolution_mm=resolution_mm,
        origin_mm=origin_mm,
        source='csv',
        metadata={"header": header_lines},
    )


def read_image_dose(filepath_or_bytes, filename=None, resolution_mm=None, max_dose_gy=None):
    """Lê imagem PNG/TIFF como mapa de dose.

    A imagem deve ser uma representação da dose (ex: heatmap exportado do TPS).
    Pixels mais claros = dose maior (ou vice-versa, dependendo da convenção).

    Args:
        resolution_mm: mm por pixel (deve ser informado pelo usuário)
        max_dose_gy: dose máxima em Gy correspondente ao pixel mais intenso
    """
    content = _read_raw(filepath_or_bytes)
    img = Image.open(io.BytesIO(content))
    img_array = np.array(img)

    # Converter para escala de cinza se for colorida
    if img_array.ndim == 3:
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]  # remover alpha
        # Usar luminância
        gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    else:
        gray = img_array.astype(np.float64)

    # Normalizar para 0-1
    gray_min, gray_max = gray.min(), gray.max()
    if gray_max > gray_min:
        normalized = (gray - gray_min) / (gray_max - gray_min)
    else:
        normalized = np.zeros_like(gray)

    # Converter para dose
    if max_dose_gy is not None:
        dose = normalized * float(max_dose_gy)
    else:
        dose = normalized  # dose relativa (0-1)

    if resolution_mm is None:
        resolution_mm = 1.0  # placeholder

    return DoseDistribution(
        dose=dose,
        resolution_mm=resolution_mm,
        origin_mm=(0.0, 0.0),
        source='image',
        metadata={
            "filename": filename,
            "max_dose_gy": max_dose_gy,
            "image_dtype": str(img_array.dtype),
        },
    )


def read_tps(filepath_or_bytes, filename=None, **kwargs):
    """Função universal: detecta formato e lê automaticamente.

    Args:
        filepath_or_bytes: arquivo ou bytes
        filename: nome do arquivo (para detectar extensão)
        **kwargs: parâmetros adicionais (resolution_mm, max_dose_gy, etc.)

    Returns:
        DoseDistribution
    """
    fmt = detect_format(filepath_or_bytes, filename)

    if fmt == 'dicom':
        return read_dicom(filepath_or_bytes, filename)
    elif fmt == 'eclipse_all':
        return read_eclipse_all(filepath_or_bytes, filename)
    elif fmt == 'csv':
        return read_csv_dose(filepath_or_bytes, filename, **kwargs)
    elif fmt == 'image':
        return read_image_dose(filepath_or_bytes, filename, **kwargs)
    else:
        raise ValueError(f"Formato não suportado: {fmt}")
