# ============================================================
# INFORME FINANCIERO MENSUAL — PORTAFOLIO Y ACTIVOS
# ============================================================
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Table, TableStyle, KeepInFrame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from scipy.stats import skew, kurtosis, jarque_bera

# ============================================================
# CONFIGURACIÓN PRINCIPAL (FECHAS AJUSTADAS A TRIMESTRAL)
# ============================================================
TICKERS = ["AAPL", "AMZN", "JPM", "GS", "SPY", "DIA", "BND"]
PESOS = [0.1311, 0.1317, 0.1318, 0.1335, 0.1595, 0.1619, 0.1478]
# ✅ FECHAS CORREGIDAS PARA ANÁLISIS TRIMESTRAL (20-08-2025 a 07-11-2025)
FECHA_INICIO = "2015-01-01"
FECHA_FIN = "2025-11-13"
VENTANA_VOL = 21

PLOT_WIDTH = 10
PLOT_HEIGHT = 9.0

# =========================================================================
# ✅ ANÁLISIS : COMENTARIOS SOBRE EVOLUCIÓN (Columna Izquierda)
# =========================================================================

comentario_descripcion = {
    "AAPL": "<b>Apple Inc. (AAPL):</b> -",
    "AMZN": "<b>Amazon Inc. (AMZN):</b> -",
    "JPM": 	"<b>JPMorgan Chase & Co. (JPM):</b>-",
    "GS": 	"<b>Goldman Sachs Group Inc. (GS):</b> -",
    "SPY": 	"<b>SPDR S&P 500 ETF Trust (SPY):</b> -",
    "DIA": 	"<b>SPDR Dow Jones Industrial Average ETF Trust (DIA):</b> -",
    "BND": 	"<b>Vanguard Total Bond Market ETF (BND):</b> -",
    "PORTaFOLIO": "<b>Análisis cuantitativo del PORTFOLIO:</b> -"
}

# =========================================================================
# ✅ ANÁLISIS: COMENTARIOS SOBRE RIESGO (Columna Derecha)
# =========================================================================
comentario_descripcion = {
    "AAPL": "<b>Apple Inc. (AAPL):</b> -",
    "AMZN": "<b>Amazon Inc. (AMZN):</b> -",
    "JPM": "<b>JPMorgan Chase & Co. (JPM):</b> -",
    "GS": "<b>Goldman Sachs Group Inc. (GS):</b> -",
    "SPY": "<b>SPDR S&P 500 ETF Trust (SPY):</b> -",
    "DIA": "<b>SPDR Dow Jones Industrial Average ETF Trust (DIA):</b> -",
    "BND": "<b>Vanguard Total Bond Market ETF (BND):</b> -",
    "PORTAFOLIO": "<b>Análisis cuantitativo del PORTFOLIO:</b> -"
}

# =========================================================================

comentario_metricas = {
    "AAPL": "<b>Comentarios sobre AAPL:</b> -",
    "AMZN": "<b>Comentarios sobre AMZN:</b> -",
    "JPM": "<b>Comentarios sobre JPM:</b> -",
    "GS": "<b>Comentarios sobre GS:</b> -",
    "SPY": "<b>Comentarios sobre SPY:</b> -",
    "DIA": "<b>Comentarios sobre DIA:</b> -",
    "BND": "<b>Comentarios sobre BND:</b> -",
    "PORTAFOLIO": "<b>Comentarios sobre PORTFOLIO:</b> -"
}

# ============================================================
# FUNCIONES DE DESCARGA Y CÁLCULO
# ============================================================
def descargar_precios(tickers, start, end):
    df_full = yf.download(tickers, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if isinstance(df_full.columns, pd.MultiIndex):
        if 'Adj Close' in df_full.columns.get_level_values(0):
            df = df_full.loc[:, "Adj Close"]
        elif 'Close' in df_full.columns.get_level_values(0):
            df = df_full.loc[:, "Close"]
        else:
            raise KeyError("No se encontró 'Adj Close' ni 'Close' en los datos multi-index.")
    elif "Adj Close" in df_full.columns:
        df = df_full["Adj Close"]
    elif "Close" in df_full.columns:
        df = df_full["Close"]
    else:
        raise KeyError("No se encontró 'Adj Close' ni 'Close' en los datos descargados.")

    if isinstance(df, pd.Series):
        if not isinstance(tickers, list):
            tickers = [tickers]
        df = df.to_frame(name=tickers[0])

    return df.dropna(how="all")

def construir_portafolio(df, pesos):
    """
    Construye el índice del portafolio, ajustando los pesos si
    algunos activos aún no tienen datos disponibles al inicio.
    """
    pesos_arr = np.array(pesos)

    # 1. Determinar el primer precio disponible (no-NaN) para cada ticker.
    precios_iniciales = df.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else np.nan, axis=0)

    # 2. Normalizar todos los precios por su primer valor no-NaN.
    df_normalizado = df / precios_iniciales

    # 3. Crear una máscara de disponibilidad (1 si hay datos, 0 si no)
    df_mask = df.notna().astype(int)

    # 4. Calcular el peso original de cada ticker solo si hay datos (0 si es NaN)
    pesos_por_fecha = df_mask * pesos_arr

    # 5. Normalizar estos pesos para que sumen 1 en cada fecha (redistribución automática)
    suma_pesos_por_fecha = pesos_por_fecha.sum(axis=1)

    # Evitar divisiones por cero
    suma_pesos_por_fecha[suma_pesos_por_fecha == 0] = 1

    pesos_ajustados = pesos_por_fecha.div(suma_pesos_por_fecha, axis=0)

    # 6. Calcular la contribución ponderada. Remplazamos NaN en precios normalizados con 0
    contribucion = df_normalizado.fillna(0).mul(pesos_ajustados)

    # 7. Sumar las contribuciones para obtener el valor del portafolio
    port = contribucion.sum(axis=1)
    port.name = "PORTAFOLIO"
    return port

def calcular_metricas(retornos, benchmark=None):
    if retornos.empty or len(retornos) < 2:
        return {k: np.nan for k in ["Beta", "Volatilidad", "Sharpe", "Sortino", "Alpha Jensen", "VaR 95", "VaR 99", "Profit", "Dias"]}

    mean_ann = retornos.mean() * 252
    std_ann = retornos.std() * np.sqrt(252)
    risk_free = 0.0377
    sharpe = (mean_ann-risk_free)/ std_ann if std_ann != 0 else np.nan

    neg_ret = retornos[retornos < 0]
    neg_std_ann = neg_ret.std() * np.sqrt(252) if not neg_ret.empty else 0
    sortino = mean_ann / neg_std_ann if neg_std_ann > 0 else np.nan

    rend_total = (np.exp(np.log1p(retornos).sum()) - 1) * 100
    VaR95 = np.percentile(retornos, 5) * 100
    VaR99 = np.percentile(retornos, 1) * 100
    dias = len(retornos)

    beta, alpha = np.nan, np.nan
    if benchmark is None:
        return {
            "Beta": 1.0,
            "Volatilidad": std_ann,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Alpha Jensen": 0.0,  # El Alpha de un índice contra sí mismo es teóricamente cero.
            "VaR 95": VaR95,
            "VaR 99": VaR99,
            "Profit": rend_total,
            "Dias": dias
        }


    if not benchmark.empty and len(benchmark) > 2:
        common_idx = retornos.index.intersection(benchmark.index)
        if len(common_idx) > 2:
            r_common = retornos.reindex(common_idx)
            b_common = benchmark.reindex(common_idx)
            cov = np.cov(r_common, b_common)[0, 1]
            var_b = np.var(b_common)
            if var_b != 0:
                beta = cov / var_b
                alpha = mean_ann - risk_free - beta * (b_common.mean() * 252 - risk_free)

    return {
        "Beta": beta,
        "Volatilidad": std_ann,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Alpha Jensen": alpha,
        "VaR 95": VaR95,
        "VaR 99": VaR99,
        "Profit": rend_total,
        "Dias": dias
    }

# ----------------------------------------------------------------------------
# FUNCIÓN: DESCARGA DE DATOS FUNDAMENTALES Y CÁLCULO DE UPGRADE (Se mantiene sin cambios)
# ----------------------------------------------------------------------------
def descargar_fundamentales(tickers):
    """Descarga información financiera clave (Price, PE, EPS, Target) para los tickers."""
    data = {}
    for ticker in tickers:
        empty_data = {
            "Price Close": "-", "P/E ratio": "-", "EPS": "-",
            "Fwd Dividend": "-", "1y Target": "-",
            "Upgrade": "-",
        }
        try:
            # El portafolio no tiene data fundamental en yfinance
            if ticker == "PORTFOLIO":
                data[ticker] = empty_data
                continue

            t = yf.Ticker(ticker)
            info = t.info

            # Obtener datos requeridos
            prev_close = info.get('previousClose')
            pe_ratio = info.get('trailingPE')
            eps = info.get('trailingEps')

            # Formateo de dividendos y rendimiento
            div_fwd = info.get('forwardAnnualDividendRate')
            yield_fwd = info.get('forwardAnnualDividendYield')
            dividend_yield = f"{div_fwd:.2f} & {yield_fwd*100:.2f}%" if div_fwd is not None and yield_fwd is not None else "-"

            # Estimación del objetivo a 1 año
            target_est = info.get('targetMedianPrice')

            # Cálculo del Upgrade (Retorno proyectado): log(1y Target Est / Previos Closes)
            upgrade = np.nan
            if prev_close is not None and target_est is not None and prev_close > 0:
                upgrade = (np.log(target_est / prev_close)) * 100

            data[ticker] = {
                "Price Close": f"${prev_close:.2f}" if prev_close is not None else "-",
                "P/E ratio": f"{pe_ratio:.2f}" if pe_ratio is not None else "-",
                "EPS": f"${eps:.2f}" if eps is not None else "-",
                "Fwd Dividend": dividend_yield,
                "1y Target": f"${target_est:.2f}" if target_est is not None else "-",
                "Upgrade": f"{upgrade:.2f}%" if not np.isnan(upgrade) else "-",
            }
        except Exception:
            data[ticker] = empty_data

    return data
# ----------------------------------------------------------------------------


# ============================================================
# FUNCIONES DE GRAFICOS (Se mantienen sin cambios)
# ============================================================
def generar_graficos(nombre, precios, rend, vol_roll, fig_width=PLOT_WIDTH, fig_height=PLOT_HEIGHT):
    fig, axs = plt.subplots(3, 1, figsize=(fig_width, fig_height))
    plt.subplots_adjust(hspace=0.45)

    axs[0].plot(precios.index, precios.values, color="navy", linewidth=1)
    axs[0].set_title("Evolución del valor ($)", fontsize=9)

    axs[1].plot(rend.index, rend.values * 100, color="gray", linewidth=1)
    axs[1].set_title("Rendimientos diarios (%)", fontsize=9)

    axs[2].plot(vol_roll.index, vol_roll.values * 100, color="orange", linewidth=1)
    axs[2].set_title("Volatilidad rolling 21d (%)", fontsize=9)

    for ax in axs:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', labelrotation=30, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

    ### INICIO DE LA PARTE VISUAL DEL PDF ####
    fig.suptitle(f"{nombre} ({FECHA_INICIO} → {FECHA_FIN})", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buffer.seek(0)
    return buffer

def generar_var_plot_con_estadisticas(retornos, nivel=5):
    datos = retornos.dropna()

    if datos.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No hay datos suficientes para VaR", ha="center", va="center")
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=200, facecolor='white')
        plt.close(fig)
        buf.seek(0)
        stats_text = "Muestras: 0 | Media: - | Mediana: - | Skew: - | Kurt: - | JB: - | p-val: -"
        return buf, stats_text

    datos_pct = datos * 100.0
    n = len(datos_pct)

    if n >= 3:
        media = datos_pct.mean()
        mediana = datos_pct.median()
        sk = skew(datos_pct)
        kt = kurtosis(datos_pct, fisher=False)
        jb_stat, jb_p = jarque_bera(datos_pct)
    else:
        media, mediana, sk, kt, jb_stat, jb_p = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    VaR_val = np.percentile(datos_pct, nivel)
    below = datos_pct[datos_pct <= VaR_val]
    CVaR_val = below.mean() if not below.empty else VaR_val

    def format_stat(val, format_str):
        return format_str.format(val) if not np.isnan(val) else '-'

    # TAMAÑO DE LA PÁGINA AJUSTADO A LOS GRAFICOS
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(datos_pct, bins=40, edgecolor='k', alpha=0.6)
    ax.axvline(VaR_val, linestyle='--', linewidth=1.5, color='red', label=f"VaR (p{nivel}) = {VaR_val:.2f}%")
    ax.axvline(CVaR_val, linestyle='-', linewidth=1.5, color='orange', label=f"CVaR = {CVaR_val:.2f}%")
    ax.set_title("Distribución de retornos — VaR histórico", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    stats_text = (
        f"Muestras: {n} | Media: {format_stat(media, '{:.4f}')}% | Mediana: {format_stat(mediana, '{:.4f}')}% | "
        f"Skew: {format_stat(sk, '{:.4f}')} | Kurt: {format_stat(kt, '{:.4f}')} | JB: {format_stat(jb_stat, '{:.4f}')} | p-val: {format_stat(jb_p, '{:.4f}')}"
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf, stats_text,

# ============================================================
# CONFIGURACION PLANTILLA PDF (HEADER/FOOTER) 
# ============================================================
COLOR_BARRA = colors.HexColor("#0B3D91")
COLOR_LINEA = colors.HexColor("#808080")
TITULO_DOC = "Informe Financiero del Portafolio"
DESCRIPCION_HEADER = "Análisis cuantitativo y desempeño de activos"
INTEGRANTES = "Grupo 1 - Equipo de Inversión"
RESUMEN_INTEGRANTES = "Autor: Carlos Caballero."
PIE_IZQ = "Fuente: Yahoo Finance. - Documento de caracter académico."
NOMBRE_ARCHIVO = "informe_cuantitativo_final.pdf" # Nombre de archivo actualizado
FECHA_EJECUCION = datetime.now().strftime("11/11/2025")

def crear_encabezado_pie(pdf, num_pagina, total_paginas):
    ancho, alto = A4
    altura_barra = 3 * cm
    pdf.setFillColor(COLOR_BARRA)
    pdf.rect(0, alto - altura_barra, ancho, altura_barra, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(1.5 * cm, alto - 1.2 * cm, TITULO_DOC)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(1.5 * cm, alto - 2.0 * cm, f"Fecha: {FECHA_EJECUCION}")
    pdf.drawString(1.5 * cm, alto - 2.5 * cm, DESCRIPCION_HEADER)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawRightString(ancho - 1.5 * cm, alto - 1.2 * cm, INTEGRANTES)
    pdf.setFont("Helvetica", 9)
    pdf.drawRightString(ancho - 1.5 * cm, alto - 2.0 * cm, RESUMEN_INTEGRANTES)
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(colors.black)
    pdf.drawString(1.5 * cm, 1 * cm, PIE_IZQ)
    pdf.drawRightString(ancho - 1.5 * cm, 1 * cm, f"Página {num_pagina} de {total_paginas}")

# ============================================================
# GENERAR PDF FINAL (FUNCIÓN CORREGIDA Y AJUSTADA) (Se mantiene sin cambios)
# ============================================================
def generar_pdf():
    # Las fechas ya están definidas globalmente

    precios = descargar_precios(TICKERS, FECHA_INICIO, FECHA_FIN)

    # Manejo de datos insuficientes
    if precios.empty or len(precios) < 3:
        print(f"\u274C Error: No hay suficientes datos (mínimo 3 días) entre {FECHA_INICIO} y {FECHA_FIN} para un análisis significativo.")
        return

    port = construir_portafolio(precios, PESOS)
    precios["PORTAFOLIO"] = port

    orden_final = TICKERS + ["PORTAFOLIO"]
    precios = precios[orden_final]

    retornos = np.log(precios / precios.shift(1)).dropna()

    # DESCARGAR DATOS FUNDAMENTALES
    tickers_fundamentales = TICKERS + ["PORTAFOLIO"]
    datos_fundamentales = descargar_fundamentales(tickers_fundamentales)

    benchmark_spy = retornos["SPY"] if "SPY" in retornos.columns else None

    total_paginas = len(precios.columns) + 1
    pdf = canvas.Canvas(NOMBRE_ARCHIVO, pagesize=A4)
    ancho, alto = A4

    #_####################### PORTADA ############################
    # ----------------- Portada -----------------
    pdf.setFillColor(COLOR_BARRA)
    pdf.rect(0, 0, ancho, alto, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawCentredString(ancho / 2, alto / 2 + 2 * cm, TITULO_DOC)
    pdf.setFont("Helvetica", 14)
    pdf.drawCentredString(ancho / 2, alto / 2 - 1 * cm, f"Fecha de generación: {FECHA_EJECUCION}")
    pdf.showPage()

    # ----------------- Páginas por activo -----------------
    for i, col in enumerate(precios.columns, start=2):
        crear_encabezado_pie(pdf, i, total_paginas)

        LEFT_MARGIN_CONTENT = 1 * cm
        RIGHT_MARGIN_CONTENT = 1 * cm
        TOP_CONTENT_AREA_Y = alto - 4.5 * cm
        BOTTOM_CONTENT_AREA_Y = 2.5 * cm

        content_area_height = TOP_CONTENT_AREA_Y - BOTTOM_CONTENT_AREA_Y
        total_content_width = ancho - LEFT_MARGIN_CONTENT - RIGHT_MARGIN_CONTENT

        LEFT_COL_RATIO = 0.7
        left_col_width = total_content_width * LEFT_COL_RATIO
        gutter_width = 1 * cm
        right_col_width = total_content_width - left_col_width - gutter_width

        left_col_start_x = LEFT_MARGIN_CONTENT
        right_col_start_x = left_col_start_x + left_col_width + gutter_width

        # Línea divisoria
        pdf.setStrokeColor(COLOR_LINEA)
        pdf.setLineWidth(1)
        pdf.line(left_col_start_x + left_col_width + gutter_width/2, BOTTOM_CONTENT_AREA_Y,
                 left_col_start_x + left_col_width + gutter_width/2, TOP_CONTENT_AREA_Y)

        # ----------------- Columna izquierda -----------------
        current_y_left = TOP_CONTENT_AREA_Y
        pdf.setFont("Helvetica-Bold", 14)
        pdf.setFillColor(colors.black)
        pdf.drawString(left_col_start_x, current_y_left, col)
        current_y_left -= 0.4 * cm

        # Descripción
        estilo_desc = ParagraphStyle("Normal", fontName="Helvetica", fontSize=10, leading=12)
        descripcion_para_text = f"<b>Análisis cuantitativo de {col}:</b> "
        desc_frame_height = 1 * cm
        desc_frame = Frame(left_col_start_x, current_y_left - desc_frame_height, left_col_width, desc_frame_height, showBoundary=0)
        desc_frame.addFromList([Paragraph(descripcion_para_text, estilo_desc)], pdf)
        current_y_left -= (desc_frame_height + 0.05 * cm)

        # Comentario descripción
        comentario_text = comentario_descripcion.get(col, "") or ""

        estilo_coment = ParagraphStyle("coment", fontName="Helvetica", fontSize=10, leading=12)
        comentario_frame_height = 4.0 * cm

        kif_desc = KeepInFrame(left_col_width, comentario_frame_height, [Paragraph(comentario_text, estilo_coment)], mode='overflow')
        comentario_frame = Frame(left_col_start_x, current_y_left - comentario_frame_height, left_col_width, comentario_frame_height, showBoundary=0)
        comentario_frame.addFromList([kif_desc], pdf)
        current_y_left -= (comentario_frame_height + 0.1 * cm)

        # ----------------- Gráficos principales -----------------
        s_price = precios[col].dropna()
        asset_ret = retornos[col].dropna()

        asset_vol = pd.Series(dtype=float)
        if not asset_ret.empty:
            asset_vol = asset_ret.rolling(VENTANA_VOL).std().dropna() * np.sqrt(252)

        # CÁLCULO DE DIMENSIONES PARA ALINEACIÓN HORIZONTAL
        final_image_width = left_col_width
        final_image_height = final_image_width * (PLOT_HEIGHT / PLOT_WIDTH)

        if not s_price.empty and len(s_price) > 1 and not asset_ret.empty:
            # Gráfico principal
            img_buffer = generar_graficos(col, s_price, asset_ret, asset_vol)
            img = ImageReader(img_buffer)

            image_height_available = current_y_left - BOTTOM_CONTENT_AREA_Y - 0.8 * cm
            target_aspect_ratio = PLOT_HEIGHT / PLOT_WIDTH

            # Recalculando el tamaño final para que quepa verticalmente
            if final_image_height > image_height_available * 0.7:
                final_image_height = image_height_available * 0.7
                final_image_width = final_image_height / target_aspect_ratio # Mantiene la proporción

            img_y_pos = current_y_left - final_image_height
            if img_y_pos < (BOTTOM_CONTENT_AREA_Y + 1.2 * cm):
                img_y_pos = BOTTOM_CONTENT_AREA_Y + 1.2 * cm

            pdf.drawImage(img, left_col_start_x, img_y_pos,
                          width=final_image_width, height=final_image_height,
                          preserveAspectRatio=True, anchor='sw')

            # Gráfico VaR + stats
            var_buffer, stats_text, = generar_var_plot_con_estadisticas(asset_ret, nivel=5)
            var_img = ImageReader(var_buffer)

            stats_height = 1.0 * cm
            var_plot_height_ratio = final_image_height / 3.0 # Altura proporcional
            var_plot_height = max(4.0 * cm, var_plot_height_ratio)
            gap_between = 0.1 * cm

            stats_y = img_y_pos - stats_height - gap_between
            var_y = stats_y - var_plot_height - gap_between

            if var_y < BOTTOM_CONTENT_AREA_Y + 0.5 * cm:
                # Si no cabe, ajustar (mantenemos la lógica de reajuste)
                available_for_var = img_y_pos - BOTTOM_CONTENT_AREA_Y - 0.5 * cm - gap_between
                if available_for_var > 0.8 * cm:
                    var_plot_height = max(0.8 * cm, available_for_var - stats_height - gap_between)
                    stats_y = img_y_pos - stats_height - gap_between
                    var_y = stats_y - var_plot_height - gap_between
                else:
                    var_plot_height = 5 * cm
                    stats_height = 0

            # Stats como párrafo centrado
            if stats_text:
                estilo_stats = ParagraphStyle("stats", fontName="Helvetica", fontSize=7, leading=10, alignment=1)
                stats_para = Paragraph(stats_text, estilo_stats)
                desplazamiento_y = 1 * cm
                stats_frame = Frame(left_col_start_x, stats_y - desplazamiento_y, left_col_width, stats_height * 2, showBoundary=0)
                stats_frame.addFromList([stats_para], pdf)

            # MODIFICACIÓN 2: Usar el mismo ancho que la imagen superior (final_image_width)
            var_img_width = final_image_width
            var_x = left_col_start_x 

            pdf.drawImage(var_img, var_x, var_y,
                            width=var_img_width, height=var_plot_height,
                            preserveAspectRatio=False, mask='auto')

            current_y_left = var_y - 0.3 * cm

        else:
            pdf.setFont("Helvetica", 10)
            pdf.drawString(left_col_start_x, current_y_left - 2 * cm, "No hay datos suficientes para generar gráficos.")
            current_y_left -= 2.2 * cm

        # ------- Columna derecha (REORGANIZADA Y CON TABLA SUMMARY ------------
        current_y_right = TOP_CONTENT_AREA_Y
        estilo_info_derecha = ParagraphStyle("Normal", fontName="Helvetica", fontSize=9, leading=11, alignment=2)
        desired_table_width_cm = 4.5 * cm
        col_width_1 = desired_table_width_cm * 0.55
        col_width_2 = desired_table_width_cm * 0.45
        table_x_pos = right_col_start_x + 0.2 * cm 

        # 1. INFO DE TEMPORALIDAD
        texto_info_para_text = f"<b>Temporalidad:</b> Trimestral<br/><b>Serie:</b> {col}"
        info_frame_derecha_height = 1.5 * cm
        info_frame_derecha = Frame(right_col_start_x, current_y_right - info_frame_derecha_height,
                                   right_col_width, info_frame_derecha_height, showBoundary=0)
        info_frame_derecha.addFromList([Paragraph(texto_info_para_text, estilo_info_derecha)], pdf)
        current_y_right -= (info_frame_derecha_height + 0.5 * cm)


        # 2. TABLA MÉTRICAS DE RIESGO
        benchmark_para_metricas = None
        if col != "SPY":
            benchmark_para_metricas = retornos["SPY"] if "SPY" in retornos.columns else None

        metrics = calcular_metricas(asset_ret, benchmark=benchmark_para_metricas)

        table_data = [["Métricas de riesgo", ""]]
        keys_order = ["Beta","Volatilidad","Sharpe","Sortino","Alpha Jensen","VaR 95","VaR 99","Profit","Dias"]
        for k in keys_order:
            v = metrics.get(k, np.nan)
            if isinstance(v, float) and not np.isnan(v):
                cell_value = f"{v:.2f}"
                if k in ["VaR 95", "VaR 99", "Profit"]:
                    cell_value += "%"
            else:
                cell_value = "" if v is None else str(v)
            table_data.append([k, cell_value])

        tbl = Table(table_data, colWidths=[col_width_1, col_width_2])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("SPAN", (0, 0), (1, 0)),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
            ("BOX", (0, 0), (-1, -1), 1, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))

        table_width, table_height = tbl.wrapOn(pdf, desired_table_width_cm, content_area_height)

        table_metrics_y_pos = current_y_right - table_height - 0.5 * cm
        if table_metrics_y_pos < BOTTOM_CONTENT_AREA_Y + 0.5 * cm:
            table_metrics_y_pos = BOTTOM_CONTENT_AREA_Y + 0.5 * cm
        tbl.drawOn(pdf, table_x_pos, table_metrics_y_pos)

        # 3. TABLA SUMMARY (Nueva, debajo de Métricas)
        summary_data = datos_fundamentales.get(col, {})
        table_summary_data = [["Summary", ""]]
        keys_order_summary = ["Price Close", "P/E ratio", "EPS", "Fwd Dividend", "1y Target", "Upgrade"]
        for k in keys_order_summary:
            v = summary_data.get(k, "-")
            table_summary_data.append([k, v])

        tbl_summary = Table(table_summary_data, colWidths=[col_width_1, col_width_2])
        tbl_summary.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")), # Color verde para distinguir
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("SPAN", (0, 0), (1, 0)),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
            ("BOX", (0, 0), (-1, -1), 1, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))

        table_summary_width, table_summary_height = tbl_summary.wrapOn(pdf, desired_table_width_cm, content_area_height)

        # Posicionar la tabla Summary debajo de la tabla Métricas
        table_summary_y_pos = table_metrics_y_pos - table_summary_height - 0.2 * cm

        tbl_summary.drawOn(pdf, table_x_pos, table_summary_y_pos)

        # 4. COMENTARIO 2 (Debajo de Summary)
        comentario2_text = comentario_metricas.get(col, "") or ""
        estilo_coment2 = ParagraphStyle("coment2", fontName="Helvetica", fontSize=10, leading=12)
        comentario2_frame_height = 4.0 * cm

        kif2 = KeepInFrame(right_col_width, comentario2_frame_height, [Paragraph(comentario2_text, estilo_coment2)], mode='overflow')

        comentario2_y = table_summary_y_pos - comentario2_frame_height - 0.2 * cm

        if comentario2_y < BOTTOM_CONTENT_AREA_Y + 0.5 * cm:
            comentario2_y = BOTTOM_CONTENT_AREA_Y + 0.5 * cm

        comentario2_frame = Frame(table_x_pos, comentario2_y, desired_table_width_cm, comentario2_frame_height, showBoundary=0)
        comentario2_frame.addFromList([kif2], pdf)
        pdf.showPage()

    pdf.save()
    print(f"\u2705 PDF generado exitosamente: {NOMBRE_ARCHIVO}")

if __name__ == "__main__":
    generar_pdf()
