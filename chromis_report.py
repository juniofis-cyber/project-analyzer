"""
chromis_report.py — Sistema de Relatórios Profissionais

Gera PDFs completos com:
- Logo Chromis, data, autor, Lattes
- Seções condicionais (só aparecem se a análise foi feita)
- Formato digno de hospital/universidade
"""

import io
import base64
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False


class ChromisReport(FPDF):
    """PDF profissional Chromis."""

    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=15)
        self.sections = []  # lista de seções a incluir

    # ==================== HEADER / FOOTER ====================

    def header(self):
        # Logo Chromis (se disponível)
        try:
            self.image("Chromis_logo_principal.png", x=10, y=8, w=25)
        except:
            pass

        # Título do relatório
        self.set_font('Helvetica', 'B', 14)
        self.set_xy(40, 10)
        self.set_text_color(30, 58, 95)  # navy #1e3a5f
        self.cell(0, 8, 'Relatorio de Dosimetria com Filmes Radiocromicos', ln=True)

        # Subtítulo
        self.set_font('Helvetica', '', 9)
        self.set_xy(40, 18)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f'Chromis v3.2 | {datetime.now().strftime("%d/%m/%Y %H:%M")}', ln=True)

        # Linha separadora
        self.set_draw_color(30, 58, 95)
        self.line(10, 26, 200, 26)
        self.ln(18)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, f'Pagina {self.page_no()}', align='C')
        self.cell(0, 5, 'Desenvolvido por MACIEL, J.O. | http://lattes.cnpq.br/3347976525922556', align='R')

    # ==================== SEÇÕES ====================

    def cover_page(self, analises_realizadas, info_paciente=None):
        """Página de capa com resumo."""
        self.add_page()

        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(30, 58, 95)
        self.cell(0, 12, 'RELATORIO DE DOSIMETRIA', ln=True, align='C')
        self.ln(5)

        # Caixa de informações
        self.set_fill_color(240, 245, 250)
        self.rect(15, 50, 180, 80, style='F')

        self.set_xy(20, 55)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(30, 58, 95)
        self.cell(0, 7, 'INFORMACOES GERAIS', ln=True)

        self.set_font('Helvetica', '', 10)
        self.set_text_color(50, 50, 50)

        infos = [
            f'Data/Hora: {datetime.now().strftime("%d/%m/%Y %H:%M")}',
            f'Software: Chromis v3.2',
            f'Autor: MACIEL, J. O.',
            f'Curriculo: http://lattes.cnpq.br/3347976525922556',
        ]
        if info_paciente:
            infos.extend([
                f'Paciente: {info_paciente.get("nome", "N/A")}',
                f'Plano: {info_paciente.get("plano", "N/A")}',
                f'Isocentro: {info_paciente.get("isocentro", "N/A")}',
            ])

        for info in infos:
            self.cell(0, 6, info, ln=True)

        # Análises realizadas
        self.ln(10)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(30, 58, 95)
        self.cell(0, 7, 'ANALISES REALIZADAS', ln=True)

        for analise in analises_realizadas:
            status = '✓' if analise.get('feita', False) else '✗'
            self.set_font('Helvetica', '', 10)
            self.cell(0, 6, f'{status} {analise.get("nome", "")}', ln=True)

    def section_calibration(self, curva_data, r2=None):
        """Seção de calibração."""
        self.add_page()
        self._section_title('1. Curva de Calibracao')

        if curva_data:
            self.set_font('Helvetica', '', 10)
            self.cell(0, 6, f'Tipo de fitting: {curva_data.get("tipo_fitting", "N/A")}', ln=True)
            if r2:
                self.cell(0, 6, f'R²: {r2:.4f}', ln=True)

            # Inserir gráfico se disponível
            if 'grafico_path' in curva_data:
                self.image(curva_data['grafico_path'], x=20, y=60, w=170)

    def section_dose_map(self, dose_map_path, max_dose=None):
        """Seção de mapa de dose."""
        self.add_page()
        self._section_title('2. Mapa de Dose 2D')

        if dose_map_path:
            self.image(dose_map_path, x=20, y=50, w=170)

        if max_dose:
            self.set_font('Helvetica', '', 10)
            self.set_xy(20, 230)
            self.cell(0, 6, f'Dose maxima: {max_dose:.2f} Gy', ln=True)

    def section_isodose(self, isodose_data):
        """Seção de isodoses."""
        self.add_page()
        self._section_title('3. Mapa de Isodoses')

        if isodose_data and 'grafico_path' in isodose_data:
            self.image(isodose_data['grafico_path'], x=20, y=50, w=170)

    def section_gamma(self, gamma_data):
        """Seção de análise gamma."""
        self.add_page()
        self._section_title('4. Analise Gamma')

        if gamma_data:
            self.set_font('Helvetica', '', 10)
            self.cell(0, 6, f'Criterio: {gamma_data.get("dd", "?")}% / {gamma_data.get("dta", "?")}mm', ln=True)
            self.cell(0, 6, f'Passing Rate: {gamma_data.get("passing_rate", 0):.1f}%', ln=True)
            self.cell(0, 6, f'Gamma medio: {gamma_data.get("gamma_mean", 0):.3f}', ln=True)
            self.cell(0, 6, f'Gamma maximo: {gamma_data.get("gamma_max", 0):.3f}', ln=True)
            self.ln(5)

            if 'grafico_path' in gamma_data:
                self.image(gamma_data['grafico_path'], x=20, y=90, w=170)

    def section_isodose_comparison(self, iso_compare_data):
        """Seção de comparação de isodoses."""
        self.add_page()
        self._section_title('5. Comparacao de Isodose — Filme vs TPS')

        if iso_compare_data:
            # Tabela de resultados
            self.set_font('Helvetica', 'B', 10)
            self.set_fill_color(230, 235, 240)
            self.cell(30, 8, 'Isodose', border=1, fill=True, align='C')
            self.cell(40, 8, 'Dose (Gy)', border=1, fill=True, align='C')
            self.cell(45, 8, 'Coincidencia (%)', border=1, fill=True, align='C')
            self.cell(45, 8, 'Dist. Media (mm)', border=1, fill=True, align='C')
            self.ln()

            for r in iso_compare_data.get('results', []):
                self.set_font('Helvetica', '', 9)
                self.cell(30, 7, f"{r.get('level', 0)}%", border=1, align='C')
                self.cell(40, 7, f"{r.get('dose_value', 0):.2f}", border=1, align='C')
                self.cell(45, 7, f"{r.get('coincidence', 0):.1f}%", border=1, align='C')
                self.cell(45, 7, f"{r.get('mean_distance_mm', 0):.2f}", border=1, align='C')
                self.ln()

            self.ln(10)
            if 'grafico_path' in iso_compare_data:
                self.image(iso_compare_data['grafico_path'], x=20, y=130, w=170)

    def summary_page(self, aprovado=True, observacoes=""):
        """Página de resumo e assinatura."""
        self.add_page()
        self._section_title('RESUMO E CONCLUSAO')

        self.set_font('Helvetica', '', 11)
        if aprovado:
            self.set_text_color(0, 128, 0)
            self.cell(0, 8, 'Status: APROVADO ✓', ln=True)
        else:
            self.set_text_color(200, 0, 0)
            self.cell(0, 8, 'Status: REPROVADO ✗', ln=True)

        self.set_text_color(50, 50, 50)
        if observacoes:
            self.ln(5)
            self.set_font('Helvetica', 'B', 10)
            self.cell(0, 7, 'Observacoes:', ln=True)
            self.set_font('Helvetica', '', 10)
            self.multi_cell(0, 6, observacoes)

        self.ln(20)
        self.set_draw_color(100, 100, 100)
        self.line(20, 200, 100, 200)
        self.set_xy(20, 203)
        self.set_font('Helvetica', '', 9)
        self.cell(80, 5, 'Assinatura do Fisico Medico', align='C')

        self.line(120, 200, 190, 200)
        self.set_xy(120, 203)
        self.cell(70, 5, 'Data', align='C')

    # ==================== HELPERS ====================

    def _section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(30, 58, 95)
        self.cell(0, 10, title, ln=True)
        self.set_draw_color(30, 58, 95)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)


def save_plot_to_file(fig, path='/tmp/chromis_plot.png', dpi=150):
    """Salva um matplotlib figure em arquivo temporário para o PDF."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def gerar_relatorio(analises, info_paciente=None, output_path='/tmp/chromis_relatorio.pdf'):
    """
    Gera o relatório completo baseado nas análises realizadas.

    analises = {
        'calibracao': {'feita': True, 'curva': {...}, 'r2': 0.999},
        'dose_map': {'feita': True, 'grafico_path': '...', 'max_dose': 10.5},
        'isodose': {'feita': True, 'grafico_path': '...'},
        'gamma': {'feita': True, 'dd': 3, 'dta': 3, 'passing_rate': 95.2, ...},
        'iso_compare': {'feita': True, 'results': [...], 'grafico_path': '...'},
    }
    """
    if not HAS_FPDF:
        raise ImportError("fpdf2 nao instalado. Instale com: pip install fpdf2")

    pdf = ChromisReport()

    # Lista de análises realizadas para a capa
    analises_lista = [
        {'nome': 'Calibracao', 'feita': analises.get('calibracao', {}).get('feita', False)},
        {'nome': 'Mapa de Dose 2D', 'feita': analises.get('dose_map', {}).get('feita', False)},
        {'nome': 'Mapa de Isodoses', 'feita': analises.get('isodose', {}).get('feita', False)},
        {'nome': 'Analise Gamma', 'feita': analises.get('gamma', {}).get('feita', False)},
        {'nome': 'Comparacao de Isodose (Filme vs TPS)', 'feita': analises.get('iso_compare', {}).get('feita', False)},
    ]

    # CAPA
    pdf.cover_page(analises_lista, info_paciente)

    # SEÇÕES CONDICIONAIS
    if analises.get('calibracao', {}).get('feita'):
        pdf.section_calibration(
            analises['calibracao'].get('curva', {}),
            analises['calibracao'].get('r2')
        )

    if analises.get('dose_map', {}).get('feita'):
        pdf.section_dose_map(
            analises['dose_map'].get('grafico_path'),
            analises['dose_map'].get('max_dose')
        )

    if analises.get('isodose', {}).get('feita'):
        pdf.section_isodose(analises['isodose'])

    if analises.get('gamma', {}).get('feita'):
        pdf.section_gamma(analises['gamma'])

    if analises.get('iso_compare', {}).get('feita'):
        pdf.section_isodose_comparison(analises['iso_compare'])

    # RESUMO
    gamma_ok = analises.get('gamma', {}).get('passing_rate', 100) >= 90
    iso_ok = all(r.get('coincidence', 0) >= 80 for r in analises.get('iso_compare', {}).get('results', []))
    aprovado = gamma_ok and iso_ok

    pdf.summary_page(aprovado=aprovado)

    pdf.output(output_path)
    return output_path
