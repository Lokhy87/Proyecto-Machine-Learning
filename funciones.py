
import matplotlib.pyplot as plt
import seaborn as sns

def visualizar_atributos_bar(ganadores, perdedores, atributos, titulo):
    fig, axes = plt.subplots(6, 2, figsize=(10, 20))
    fig.suptitle(f'Comparación de Atributos entre Ganadores y Perdedores ({titulo})', fontsize=16)
    
    for i, atributo in enumerate(atributos):
        mean_ganadores = ganadores[atributo].mean()
        mean_perdedores = perdedores[atributo].mean()
        
        sns.barplot(x=['Ganadores', 'Perdedores'], y=[mean_ganadores, mean_perdedores], ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f'{atributo}')
        axes[i//2, i%2].set_ylabel('Media')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
