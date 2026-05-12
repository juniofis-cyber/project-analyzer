"""
chromis_report.py - Relatorio PDF usando matplotlib (ja instalado)

Elimina dependencia do fpdf2. Usa matplotlib.backends.backend_pdf
que ja vem com o matplotlib (presente no requirements.txt).
"""

import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def gerar_relatorio(analises, info_paciente=None, output_path='/tmp/chromis_relatorio.pdf'):
    """Gera relatorio PDF profissional usando matplotlib."""
    
    with PdfPages(output_path) as pdf:
        
        # ===== PAGINA 1: CAPA =====
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
        ax.set_xlim(0, 8.27)
        ax.set_ylim(0, 11.69)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        # Tentar colocar logo
        try:
            from PIL import Image as PILImage
            logo = PILImage.open('Chromis_logo_principal.png')
            new_w = 1.5
            aspect = logo.height / logo.width
            new_h = new_w * aspect
            ax.imshow(logo, extent=[0.5, 0.5+new_w, 10.5-new_h, 10.5], aspect='auto')
        except:
            pass
        
        # Titulo
        ax.text(4.1, 9.5, 'RELATORIO DE DOSIMETRIA', 
                fontsize=22, fontweight='bold', ha='center', color='#1e3a5f')
        ax.text(4.1, 9.0, 'com Filmes Radiocromicos EBT3/EBT4',
                fontsize=14, ha='center', color='#64748b')
        
        # Caixa de informacoes
        rect = plt.Rectangle((1, 5.5), 6.27, 2.5, fill=True, 
                              facecolor='#f1f5f9', edgecolor='#1e3a5f', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(1.3, 7.5, 'INFORMACOES GERAIS', fontsize=12, fontweight='bold', color='#1e3a5f')
        
        y_info = 7.0
        infos = [
            f'Data/Hora: {datetime.now().strftime("%d/%m/%Y %H:%M")}',
            'Software: Chromis v3.2',
            'Autor: MACIEL, J. O.',
            'Curriculo: http://lattes.cnpq.br/3347976525922556',
        ]
        if info_paciente:
            infos.append(f'Paciente: {info_paciente.get("nome", "N/A")}')
            infos.append(f'Plano: {info_paciente.get("plano", "N/A")}')
        
        for info in infos:
            ax.text(1.3, y_info, info, fontsize=10, color='#374151')
            y_info -= 0.3
        
        # Analises realizadas
        ax.text(1.3, 4.5, 'ANALISES REALIZADAS', fontsize=12, fontweight='bold', color='#1e3a5f')
        
        y_anal = 4.0
        analises_lista = [
            ('Calibracao', analises.get('calibracao', {}).get('feita', False)),
            ('Mapa de Dose 2D', analises.get('dose_map', {}).get('feita', False)),
            ('Mapa de Isodoses', analises.get('isodose', {}).get('feita', False)),
            ('Analise Gamma', analises.get('gamma', {}).get('feita', False)),
            ('Comparacao de Isodose', analises.get('iso_compare', {}).get('feita', False)),
        ]
        
        for nome, feita in analises_lista:
            status = '[OK]' if feita else '[  ]'
            color = '#059669' if feita else '#9ca3af'
            ax.text(1.3, y_anal, f'{status} {nome}', fontsize=10, color=color)
            y_anal -= 0.3
        
        # Rodape
        ax.text(4.1, 1.0, 'Desenvolvido por MACIEL, J. O. | Chromis v3.2',
                fontsize=8, ha='center', color='#9ca3af', style='italic')
        
        pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ===== PAGINA 2: CALIBRACAO =====
        if analises.get('calibracao', {}).get('feita'):
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.set_xlim(0, 8.27)
            ax.set_ylim(0, 11.69)
            ax.axis('off')
            fig.patch.set_facecolor('white')
            
            ax.text(0.5, 10.8, '1. Curva de Calibracao', 
                    fontsize=16, fontweight='bold', color='#1e3a5f')
            ax.plot([0.5, 7.77], [10.5, 10.5], color='#1e3a5f', linewidth=2)
            
            curva = analises['calibracao'].get('curva', {})
            r2 = analises['calibracao'].get('r2', 0)
            
            ax.text(0.5, 9.8, f'Tipo de fitting: {curva.get("tipo_fitting", "N/A")}', fontsize=11)
            ax.text(0.5, 9.4, f'R2: {r2:.4f}', fontsize=11)
            ax.text(0.5, 8.8, '[Grafico de calibracao gerado na sessao]', 
                    fontsize=10, color='#9ca3af', style='italic')
            
            pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # ===== PAGINA 3: MAPA DE DOSE =====
        if analises.get('dose_map', {}).get('feita'):
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.set_xlim(0, 8.27)
            ax.set_ylim(0, 11.69)
            ax.axis('off')
            fig.patch.set_facecolor('white')
            
            ax.text(0.5, 10.8, '2. Mapa de Dose 2D', 
                    fontsize=16, fontweight='bold', color='#1e3a5f')
            ax.plot([0.5, 7.77], [10.5, 10.5], color='#1e3a5f', linewidth=2)
            
            max_dose = analises['dose_map'].get('max_dose', 0)
            ax.text(0.5, 9.8, f'Dose maxima: {max_dose:.2f} Gy', fontsize=11)
            ax.text(0.5, 9.2, '[Mapa de dose gerado na sessao]', 
                    fontsize=10, color='#9ca3af', style='italic')
            
            pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # ===== PAGINA 4: ANALISE GAMMA =====
        if analises.get('gamma', {}).get('feita'):
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.set_xlim(0, 8.27)
            ax.set_ylim(0, 11.69)
            ax.axis('off')
            fig.patch.set_facecolor('white')
            
            ax.text(0.5, 10.8, '3. Analise Gamma', 
                    fontsize=16, fontweight='bold', color='#1e3a5f')
            ax.plot([0.5, 7.77], [10.5, 10.5], color='#1e3a5f', linewidth=2)
            
            gg = analises['gamma']
            y_pos = 9.8
            ax.text(0.5, y_pos, f'Criterio: {gg.get("dd", "?")}% / {gg.get("dta", "?")}mm', fontsize=11)
            y_pos -= 0.4
            
            passing = gg.get('passing_rate', 0)
            color = '#059669' if passing >= 90 else '#dc2626'
            ax.text(0.5, y_pos, f'Passing Rate: {passing:.1f}%', fontsize=11, 
                    color=color, fontweight='bold')
            y_pos -= 0.4
            
            ax.text(0.5, y_pos, f'Gamma medio: {gg.get("gamma_mean", 0):.3f}', fontsize=11)
            y_pos -= 0.4
            ax.text(0.5, y_pos, f'Gamma maximo: {gg.get("gamma_max", 0):.3f}', fontsize=11)
            
            pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # ===== PAGINA 5: COMPARACAO DE ISODOSE =====
        if analises.get('iso_compare', {}).get('feita'):
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.set_xlim(0, 8.27)
            ax.set_ylim(0, 11.69)
            ax.axis('off')
            fig.patch.set_facecolor('white')
            
            ax.text(0.5, 10.8, '4. Comparacao de Isodose', 
                    fontsize=16, fontweight='bold', color='#1e3a5f')
            ax.plot([0.5, 7.77], [10.5, 10.5], color='#1e3a5f', linewidth=2)
            
            # Tabela
            results = analises['iso_compare'].get('results', [])
            y_pos = 9.8
            
            ax.text(0.5, y_pos, 'Isodose    Dose (Gy)    Coincidencia    Dist. Media', 
                    fontsize=10, fontweight='bold', color='#1e3a5f')
            y_pos -= 0.3
            ax.plot([0.5, 7.5], [y_pos, y_pos], color='#cbd5e1', linewidth=1)
            y_pos -= 0.3
            
            for r in results:
                level = r.get('level', 0)
                dose_val = r.get('dose_value', 0)
                coincidence = r.get('coincidence', 0)
                mean_dist = r.get('mean_distance_mm', 0)
                
                ax.text(0.5, y_pos, f'{level}%', fontsize=9)
                ax.text(2.0, y_pos, f'{dose_val:.2f}', fontsize=9)
                
                coin_color = '#059669' if coincidence >= 90 else '#dc2626'
                ax.text(3.5, y_pos, f'{coincidence:.1f}%', fontsize=9, color=coin_color)
                ax.text(5.5, y_pos, f'{mean_dist:.2f} mm', fontsize=9)
                
                y_pos -= 0.35
            
            pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # ===== PAGINA FINAL: RESUMO =====
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.set_xlim(0, 8.27)
        ax.set_ylim(0, 11.69)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        ax.text(0.5, 10.8, 'RESUMO E CONCLUSAO', 
                fontsize=16, fontweight='bold', color='#1e3a5f')
        ax.plot([0.5, 7.77], [10.5, 10.5], color='#1e3a5f', linewidth=2)
        
        gamma_ok = analises.get('gamma', {}).get('passing_rate', 100) >= 90
        iso_ok = all(r.get('coincidence', 0) >= 80 for r in analises.get('iso_compare', {}).get('results', []))
        aprovado = gamma_ok and iso_ok
        
        if aprovado:
            ax.text(0.5, 9.5, 'Status: APROVADO', 
                    fontsize=18, fontweight='bold', color='#059669')
        else:
            ax.text(0.5, 9.5, 'Status: REPROVADO', 
                    fontsize=18, fontweight='bold', color='#dc2626')
        
        # Linhas de assinatura
        ax.plot([0.5, 4.0], [3.0, 3.0], color='#374151', linewidth=1)
        ax.text(0.5, 2.7, 'Assinatura do Fisico Medico', fontsize=9, color='#6b7280')
        
        ax.plot([4.5, 7.5], [3.0, 3.0], color='#374151', linewidth=1)
        ax.text(4.5, 2.7, 'Data', fontsize=9, color='#6b7280')
        
        pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return output_path
