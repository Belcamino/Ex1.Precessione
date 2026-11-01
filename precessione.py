import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


G = 6.67430e-11
C = 2.99792458e8
M_SOLE = 1.98892e30
GM = G * M_SOLE


A_MERCURIO = 5.7909e10
E_MERCURIO = 0.2056


P_ORBITA = A_MERCURIO * (1 - E_MERCURIO**2)


FATTORE_SCALA = 1e6


N_ORBITE = 4
PUNTI_PER_ORBITA = 2000


def equazione_orbita(phi, stato):
    u, du_dphi = stato
    d2u_dphi2 = (1.0 / P_ORBITA) - u + FATTORE_SCALA * (3.0 * GM / C**2) * u**2
    return [du_dphi, d2u_dphi2]


r_perielio = A_MERCURIO * (1 - E_MERCURIO)
u0 = 1.0 / r_perielio
du0 = 0.0

stato_iniziale = [u0, du0]
phi_span = (0, N_ORBITE * 2 * np.pi)
phi_eval = np.linspace(*phi_span, N_ORBITE * PUNTI_PER_ORBITA)


soluzione = solve_ivp(
    equazione_orbita,
    phi_span,
    stato_iniziale,
    t_eval=phi_eval,
    method='RK45',
    rtol=1e-10,
    atol=1e-12
)

phi = soluzione.t
u = soluzione.y[0]
r = 1.0 / u


x = r * np.cos(phi)
y = r * np.sin(phi)


precessione_reale_rad_per_orbita = 6 * np.pi * GM / (C**2 * P_ORBITA)
precessione_reale_arcsec_per_orbita = np.degrees(precessione_reale_rad_per_orbita) * 3600


periodo_mercurio_anni = np.sqrt(A_MERCURIO**3 / GM) * 2 * np.pi / (3600*24*365.25)
orbite_per_secolo = 100 / periodo_mercurio_anni
precessione_reale_arcsec_per_secolo = precessione_reale_arcsec_per_orbita * orbite_per_secolo

print("=== Precessione reale del perielio di Mercurio (senza fattore di scala) ===")
print(f"Periodo orbitale: {periodo_mercurio_anni:.3f} anni")
print(f"Precessione per orbita: {precessione_reale_arcsec_per_orbita:.5f} arcsec")
print(f"Precessione per secolo: {precessione_reale_arcsec_per_secolo:.2f} arcsec/secolo")
print("(Il valore storicamente noto, confermato da Einstein nel 1915, è ~43 arcsec/secolo)")
print()
print(f"Fattore di scala usato nella simulazione grafica: {FATTORE_SCALA:.0e}")
print(f"Precessione 'amplificata' per orbita nel grafico: "
      f"{np.degrees(precessione_reale_rad_per_orbita * FATTORE_SCALA):.2f} gradi")


import plotly.graph_objects as go


N_FRAME = 240
indici_frame = np.linspace(0, len(x) - 1, N_FRAME).astype(int)


DURATA_FRAME_LENTO_MS = 120
DURATA_FRAME_NORMALE_MS = 40
DURATA_FRAME_VELOCE_MS = 10


frames = []
for k in indici_frame:
    frames.append(go.Frame(
        data=[
            go.Scatter(x=[0], y=[0], mode='markers',
                       marker=dict(color='orange', size=22), name='Sole'),
            go.Scatter(x=x[:k+1], y=y[:k+1], mode='lines',
                       line=dict(color='royalblue', width=1.5), name='Orbita percorsa'),
            go.Scatter(x=[x[k]], y=[y[k]], mode='markers',
                       marker=dict(color='crimson', size=10), name='Mercurio')
        ],
        name=str(k)
    ))


fig = go.Figure(
    data=[
        go.Scatter(x=[0], y=[0], mode='markers',
                   marker=dict(color='orange', size=22), name='Sole'),
        go.Scatter(x=x[:1], y=y[:1], mode='lines',
                   line=dict(color='royalblue', width=1.5), name='Orbita percorsa'),
        go.Scatter(x=[x[0]], y=[y[0]], mode='markers',
                   marker=dict(color='crimson', size=10), name='Mercurio')
    ],
    frames=frames
)


margine = 0.1
estensione = max(np.abs(x).max(), np.abs(y).max())
limite_min = -estensione * (1 + margine)
limite_max = estensione * (1 + margine)

def pulsante_play(durata_ms, etichetta):
    """Crea un pulsante che avvia l'animazione con una data velocità."""
    return dict(
        label=etichetta,
        method='animate',
        args=[None, dict(frame=dict(duration=durata_ms, redraw=True),
                          fromcurrent=True, mode='immediate',
                          transition=dict(duration=0))]
    )

pulsante_pausa = dict(
    label='⏸ Pausa',
    method='animate',
    args=[[None], dict(frame=dict(duration=0, redraw=False),
                        mode='immediate', transition=dict(duration=0))]
)

fig.update_layout(
    title=f'Precessione del perielio di Mercurio (animata)<br>'
          f'<sup>fattore di scala relativistico = {FATTORE_SCALA:.0e}</sup>',
    xaxis_title='x [m]',
    yaxis_title='y [m]',
    width=750, height=750,


    xaxis=dict(range=[limite_min, limite_max], scaleanchor='y', scaleratio=1,
               autorange=False, constrain='domain'),
    yaxis=dict(range=[limite_min, limite_max], autorange=False),
    uirevision='fisso',
    updatemenus=[dict(
        type='buttons',
        direction='left',
        x=0.0, y=1.12, xanchor='left', yanchor='top',
        buttons=[
            pulsante_play(DURATA_FRAME_LENTO_MS, '🐢 Lento (0.5x)'),
            pulsante_play(DURATA_FRAME_NORMALE_MS, '▶ Play (1x)'),
            pulsante_play(DURATA_FRAME_VELOCE_MS, '🚀 Veloce (3x)'),
            pulsante_pausa,
        ]
    )],
    sliders=[dict(
        steps=[dict(method='animate',
                    args=[[str(k)], dict(mode='immediate',
                                          frame=dict(duration=0, redraw=True),
                                          transition=dict(duration=0))],
                    label='')
               for k in indici_frame],
        x=0, y=-0.05, len=1.0,
        currentvalue=dict(prefix='Fotogramma: ')
    )]
)

fig.show()
