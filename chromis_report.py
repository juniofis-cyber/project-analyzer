"""
chromis_report.py - Sistema de Relatorios Profissionais

Gera PDFs completos com logo, data, autor, Lattes.
Sem caracteres Unicode - compativel com fpdf2/Helvetica.
"""

import io
from datetime import datetime

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

    def header(self):
        try:
            self.image("Chromis_logo_principal.png", x=10, y=8, w=25)
        except:
            pass
        self.set_font('Helvetica', 'B', 14)
        self.set_xy(40, 10)
        self.set_text_color(30, 58, 95)
        self.cell(0, 8, 'Relatorio de Dosimetria com Filmes Radiocromicos', ln=True)
        self.set_font('Helvetica', '', 9)
        self.set_xy(40, 18)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'Chromis v3.2 | ' + datetime.now().strftime("%d/%m/%Y %H:%M"), ln=True)
        self.set_draw_color(30, 58, 95)
        self.line(10, 26, 200, 26)
        self.ln(18)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, 'Pagina ' + str(self.page_no()), align='C')

    def cover_page(self, analises_lista, info_paciente=None):
        self.add_page()
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(30, 58, 95)
        self.cell(0, 12, 'RELATORIO DE DOSIMETRIA', ln=True, align='C')
        self.ln(5)

        self.set_fill_color(240, 245, 250)
        self.rect(15, 50, 180, 80, style='F')

        self.set_xy(20, 55)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(30, 58, 95)
        self.cell(0, 7, 'INFORMACOES GERAIS', ln=True)

        self.set_font('Helvetica', '', 10)
        self.set_text_color(50, 50, 50)

        infos = [
            'Data/Hora: ' + datetime.now().strftime("%d/%m/%Y %H:%M"),
            'Software: Chromis v3.2',
            'Autor: MACIEL, J. O.',
            'Curriculo: http://lattes.cnpq.br/3347976525922556',
        ]
        if info_paciente:
            infos.append('Paciente: ' + info_paciente.get("nome", "N/A"))
            infos.append('Plano: ' + info_paciente.get("plano", "N/A"))

        for info in infos:
            self.cell(0, 6, info, ln=True)

        self.ln(10)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(30, 58, 95)
        self.cell(0, 7, 'ANALISES REALIZADAS', ln=True)

        for analise in analises_lista:
            status = '[OK]' if analise.get('feita', False) else '[  ]'
            self.set_font('Helvetica', '', 10)
            self.cell(0, 6, status + ' ' + analise.get("nome", ""), ln=True)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(30, 58, 95)
        self.cell(0, 10, title, ln=True)
        self.set_draw_color(30, 58, 95)
        y = self.get_y()
        self.line(10, y, 200, y)
        self.ln(3)

    def section_calibration(self, curva_data, r2=None):
        self.add_page()
        self.section_title('1. Curva de Calibracao')
        self.set_font('Helvetica', '', 10)
        if curva_data:
            self.cell(0, 6, 'Tipo: ' + curva_data.get("tipo_fitting", "N/A"), ln=True)
        if r2:
            self.cell(0, 6, 'R2: ' + str(round(r2, 4)), ln=True)
        self.cell(0, 6, '[Grafico de calibracao - inserir imagem]', ln=True)

    def section_dose_map(self, max_dose=None):
        self.add_page()
        self.section_title('2. Mapa de Dose 2D')
        if max_dose:
            self.set_font('Helvetica', '', 10)
            self.cell(0, 6, 'Dose maxima: ' + str(round(max_dose, 2)) + ' Gy', ln=True)
        self.cell(0, 6, '[Mapa de dose - inserir imagem]', ln=True)

    def section_isodose(self):
        self.add_page()
        self.section_title('3. Mapa de Isodoses')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 6, '[Mapa de isodoses - inserir imagem]', ln=True)

    def section_gamma(self, gamma_data):
        self.add_page()
        self.section_title('4. Analise Gamma')
        self.set_font('Helvetica', '', 10)
        if gamma_data:
            self.cell(0, 6, 'Criterio: ' + str(gamma_data.get("dd", "?")) + '% / ' + str(gamma_data.get("dta", "?")) + 'mm', ln=True)
            self.cell(0, 6, 'Passing Rate: ' + str(round(gamma_data.get("passing_rate", 0), 1)) + '%', ln=True)
            self.cell(0, 6, 'Gamma medio: ' + str(round(gamma_data.get("gamma_mean", 0), 3)), ln=True)
            self.cell(0, 6, 'Gamma maximo: ' + str(round(gamma_data.get("gamma_max", 0), 3)), ln=True)
        self.cell(0, 6, '[Mapa gamma - inserir imagem]', ln=True)

    def section_iso_compare(self, iso_data):
        self.add_page()
        self.section_title('5. Comparacao de Isodose')
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(230, 235, 240)
        self.cell(30, 8, 'Isodose', border=1, fill=True, align='C')
        self.cell(40, 8, 'Dose (Gy)', border=1, fill=True, align='C')
        self.cell(45, 8, 'Coincid. (%)', border=1, fill=True, align='C')
        self.cell(45, 8, 'Dist. (mm)', border=1, fill=True, align='C')
        self.ln()
        for r in iso_data.get('results', []):
            self.set_font('Helvetica', '', 9)
            self.cell(30, 7, str(r.get('level', 0)) + '%', border=1, align='C')
            self.cell(40, 7, str(round(r.get('dose_value', 0), 2)), border=1, align='C')
            self.cell(45, 7, str(round(r.get('coincidence', 0), 1)) + '%', border=1, align='C')
            self.cell(45, 7, str(round(r.get('mean_distance_mm', 0), 2)), border=1, align='C')
            self.ln()

    def summary_page(self, aprovado=True):
        self.add_page()
        self.section_title('RESUMO E CONCLUSAO')
        self.set_font('Helvetica', '', 11)
        if aprovado:
            self.set_text_color(0, 128, 0)
            self.cell(0, 8, 'Status: APROVADO [OK]', ln=True)
        else:
            self.set_text_color(200, 0, 0)
            self.cell(0, 8, 'Status: REPROVADO [FALHA]', ln=True)
        self.set_text_color(50, 50, 50)
        self.ln(20)
        self.set_draw_color(100, 100, 100)
        self.line(20, 200, 100, 200)
        self.set_xy(20, 203)
        self.set_font('Helvetica', '', 9)
        self.cell(80, 5, 'Assinatura do Fisico Medico', align='C')
        self.line(120, 200, 190, 200)
        self.set_xy(120, 203)
        self.cell(70, 5, 'Data', align='C')


def gerar_relatorio(analises, info_paciente=None, output_path='/tmp/chromis_relatorio.pdf'):
    if not HAS_FPDF:
        raise ImportError("fpdf2 nao instalado. Instale com: pip install fpdf2")

    pdf = ChromisReport()

    analises_lista = [
        {'nome': 'Calibracao', 'feita': analises.get('calibracao', {}).get('feita', False)},
        {'nome': 'Mapa de Dose 2D', 'feita': analises.get('dose_map', {}).get('feita', False)},
        {'nome': 'Mapa de Isodoses', 'feita': analises.get('isodose', {}).get('feita', False)},
        {'nome': 'Analise Gamma', 'feita': analises.get('gamma', {}).get('feita', False)},
        {'nome': 'Comparacao de Isodose', 'feita': analises.get('iso_compare', {}).get('feita', False)},
    ]

    pdf.cover_page(analises_lista, info_paciente)

    if analises.get('calibracao', {}).get('feita'):
        pdf.section_calibration(
            analises['calibracao'].get('curva', {}),
            analises['calibracao'].get('r2')
        )

    if analises.get('dose_map', {}).get('feita'):
        pdf.section_dose_map(analises['dose_map'].get('max_dose'))

    if analises.get('isodose', {}).get('feita'):
        pdf.section_isodose()

    if analises.get('gamma', {}).get('feita'):
        pdf.section_gamma(analises['gamma'])

    if analises.get('iso_compare', {}).get('feita'):
        pdf.section_iso_compare(analises['iso_compare'])

    gamma_ok = analises.get('gamma', {}).get('passing_rate', 100) >= 90
    iso_ok = all(r.get('coincidence', 0) >= 80 for r in analises.get('iso_compare', {}).get('results', []))
    aprovado = gamma_ok and iso_ok

    pdf.summary_page(aprovado=aprovado)

    pdf.output(output_path)
    return output_path
