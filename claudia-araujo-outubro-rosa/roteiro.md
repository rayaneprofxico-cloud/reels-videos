# Reels: Vereadora Cláudia Araújo, Outubro Rosa

**Vídeo final:** `reels_claudia_araujo.mp4`, 1min18s, 1080x1920 (vertical), 30 fps, H.264 + AAC, áudio a -14 LUFS.
**Origem:** discurso de 6min34s ("fala claudia editar.mp4", Google Drive).
**Instagram:** [@v.claudiaaraujo](https://instagram.com/v.claudiaaraujo)

## Tema
Feminicídio e violência contra a mulher: "a gente precisa falar sim".

## Gancho (0–3s)
> "Dizem pra mim assim: não pode falar sobre feminicídio, porque a gente incentiva o agressor. Mas que palhaçada! A gente precisa falar sim."

Na tela: **NÃO PODE FALAR / SOBRE FEMINICÍDIO?** e logo depois **VEREADORA CLÁUDIA ARAÚJO**.

## Enredo (trechos do original remontados)

| # | Bloco | Original | Fala |
|---|---|---|---|
| 1 | Gancho | 2:45–2:55 | "Dizem pra mim… Mas que palhaçada! A gente precisa falar sim" |
| 2 | Choque | 2:31–2:40 | "A gente teve agora os 62 feminicídios… as outras mulheres estão morrendo" |
| 3 | Por que falar | 2:55–3:05 | "Porque a gente precisa mostrar pra essas mulheres que elas podem gritar, falar, serem ouvidas e serem acolhidas" |
| 4 | A realidade | 3:08–3:21 | "Mais delegacias da mulher… mais casas que acolham… passa a vida apanhando porque não tem pra onde ir" |
| 5 | Solução | 4:11–4:19 | "Eu tenho um projeto que cria microcrédito para mulheres vítimas de violência" |
| 6 | Como | 4:36–4:46 | "Como é que a gente tira as mulheres da violência? Oportunizando, dando a mão…" |
| 7 | Missão | 0:44–0:49 | "Esse é o meu papel na política: fazer com que as mulheres tenham vez e tenham voz" |
| 8 | Promessa | 5:37–5:46 | "Uma pessoa na Câmara de Vereadores que está sempre de portas abertas, pra ouvir, para acolher" |
| 9 | Final | 6:14–6:31 | "Contem com esta amiga, esta vereadora, esta empreendedora… um mundo bem mais cor-de-rosa. Obrigada." |

Pausas longas e hesitações ("éé…") foram removidas.

## Edição
- **Cores:** rosa choque `#FF2D95` (o mesmo da bandana), rosa claro, branco e vinho `#4A0626`. Leve degradê rosa nas bordas.
- **Legendas dinâmicas:** fonte Inter Black, a palavra falada acende em rosa. As palavras-chave aparecem sozinhas numa caixa rosa com "pop".
- **Destaques no topo:** gancho, nome da vereadora (no início e no bloco 8), "+ DELEGACIAS DA MULHER / + CASAS DE ACOLHIMENTO", "PROJETO DE MICROCRÉDITO", "MEU PAPEL NA POLÍTICA" e o número **62** gigante.
- **Zoom em cada corte** (100% / 115% / 130%), centralizado no rosto, para esconder as emendas e dar ritmo.
- **Flash rosa** na virada de cada bloco. **Barra de progresso** rosa no topo. Selo **lacinho + OUTUBRO ROSA**.
- **Final:** a tela vai ficando rosa em "cor-de-rosa". Depois entra o cartão "Conte com a Vereadora Cláudia Araújo · @v.claudiaaraujo · Siga ♥ Compartilhe".
- Sem música: colocar um áudio em alta no próprio Instagram, com volume baixo.

## Áudio
Corte de graves abaixo de 90 Hz (ronco e vento), redução de ruído com RNNoise e FFT, -3 dB em 250 Hz (som abafado), +3 dB em 3,5 kHz (presença), de-esser, compressor, ganho até -14 LUFS e limitador em -1,5 dBTP. Fades curtos em cada corte para evitar estalos.

## Pasta `edicao/`
- `build.py`: monta tudo com ffmpeg (cortes, zoom no rosto, áudio, legendas, gráficos e render).
- `legendas.ass`: legendas e cartões (libass).
- `palavras_sincronizadas.json`: palavras com o tempo no Reels.
- `transcricao_completa.json`: transcrição do discurso inteiro (Whisper medium), com tempo por palavra.
