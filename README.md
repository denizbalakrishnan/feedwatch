# The feed that never watched a video

**Live:** https://denizbalakrishnan.github.io/feedwatch/

A short-video feed has never watched a single one of its videos. It cannot see a dance, hear a joke or read a caption. All it is given is a grid of numbers: who watched what, and for how long. From that grid alone it gives every person and every video a short list of numbers, and those lists are enough to decide what plays next.

This project is a working version of that mechanism that you can take apart. It runs entirely in your browser: 300 simulated people, 480 videos, about 130,000 watch times, trained in under a second in the tab you are looking at.

This README is a first draft. The project is still being built for the Global Virtual Hackathon 2026 (theme: Unveiling Technology).

## What you can do with it

- **Swipe and like.** Every reaction is sent to the model as one number (the share of the video you watched). Your dot on the taste map moves straight away.
- **Reveal what the model never saw.** One switch colours every video by its topic. The topics form clear groups, although the model was never told a single topic.
- **Switch exploration on and off.** With it off, a fan of one topic is sealed into a bubble within 20 swipes. Thompson sampling keeps showing them things the model is unsure about.
- **Change what counts as evidence.** Likes only, watch time only, or both.
- **Change the crowd and the model, and retrain.** Fewer people, more noise, more gaps, 2 to 16 numbers per video. A chart shows the error on cells the model trained on and on cells hidden from it, and a score marks the map against the hidden topics.
- **Upload a brand-new video.** The first model cannot place a video nobody has watched. A second, modern kind of model (a two-tower neural network, trained in your browser with backpropagation we wrote by hand) places it at once from its tags. Both models are marked on the same hidden cells.

## How it works

| Piece | What it does | Where |
|---|---|---|
| Watch-ratio grid | The only input: one row per person, one column per video, each cell = time watched / video length | `js/sim.js` |
| Preference and confidence | Turns each cell into "yes or no" and "how sure" | `preferenceOf`, `confidenceOf` in `js/mf.js` |
| Alternating least squares | Finds a list of numbers for every person and video, plus one "appeal" number per video | `trainALS` in `js/mf.js` |
| Fold-in | Solves your own list of numbers from your reactions, after every swipe | `foldInUser` in `js/mf.js` |
| Score | appeal + (your numbers x the video's numbers, added up); the highest unseen score plays next | `scoreItems` in `js/mf.js` |
| The map | Flattens the lists to two dimensions | `principalAxes`, `procrustes2D` in `js/linalg.js` |
| Exploration | Epsilon-greedy and Thompson sampling | `js/explore.js` |
| Two-tower model | Computes a video's numbers from its ID and tags, and a person's from the videos they liked; forward pass, backward pass and Adam optimiser written by hand | `js/twotower.js` |
| Training thread | Builds the crowd and trains both models off the page's main thread | `js/train.worker.js` |

## What is computed and what is drawn by hand

Computed in your browser, from the grid and from your reactions: the crowd, the model, your vector, every score, the position of every dot, the bubble meter, the error chart and the recovery score. Nothing is pre-trained and nothing is fetched after the page loads.

Drawn by hand: the card art, titles and icons (made up from each video's topic, and never shown to the model); the overall size of your dot's movement relative to the videos (the model fixes only the product of the two lists, not the size of each, so we draw both at the same scale).

## What we simplified, and what did not work

- **The crowd is simulated.** We tried the real KuaiRec dataset first. In its fully observed part, the typical viewer watches about 7.5 seconds of every video whatever its length, so watch time says almost nothing about topic there, and the map did not sort videos by category (0.14 against 0.10 for pure chance). We would rather show a simulation that says it is one than a real-data picture that does not show what we claim. More on this will be added here.
- Real platforms use many more signals, far bigger models and several ranking stages.
- The two-tower model reads one tag per video (its topic). Real ones read the picture, the sound and the caption. Ours also treats other videos in a training batch as wrong answers without correcting for popular videos turning up more often, which the 2019 paper does correct.
- With no history at all, the two-tower model gives everybody the same single guess, which tends to be one topic. The first model falls back on general appeal and gives a mix.
- A "like" is counted as watching the video twice over.

## Things we didn't make ourselves

| What | Where used | Why we chose it |
|---|---|---|
| The implicit-feedback model and alternating least squares (Hu, Koren & Volinsky, 2008) | `js/mf.js` | The standard method for learning from watch behaviour rather than ratings |
| The per-item bias term (Koren, Bell & Volinsky, 2009) | `js/mf.js` | Separates "everyone watches this" from "you in particular would" |
| The two-tower design with in-batch sampled softmax (Covington et al., 2016; Yi et al., 2019) | `js/twotower.js` | It is how large feeds place brand-new videos |
| The Adam optimiser (Kingma & Ba, 2015) | `adamStep` in `js/twotower.js` | Trains reliably without hand-tuning a step size for every number |
| mulberry32 random number recipe (Tommy Ettinger, public domain) | `js/random.js` | A few lines, repeatable from a seed |
| Okabe-Ito colour palette | `js/ui.js` | Stays distinct for colour-blind viewers |
| Claude Code (AI coding tool) | Throughout | Used to write code and tests, which we then read, questioned and learned to explain |

No libraries, frameworks or build tools are used by the site.

## Running it

```
python -m http.server 8123
```

Then open http://localhost:8123. To run the tests: `node --test`.

## Licence

Code: MIT (see `LICENSE`).
