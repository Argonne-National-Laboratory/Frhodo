# First-Time Use

A short walkthrough of loading the bundled example and running an optimization.
Paths below are relative to the repository's `example/` folder.

## Loading an analysis (Files tab)

1. Set the **Experiment Directory** to `example/experiment`.
   - Some data should appear on screen.
2. Set the **Mechanism Directory** to `example/mechanism`.
   - A blue line appears — the initial simulated data.
3. Set the **Simulation Directory** to `example/simulation`.
   - Nothing happens yet; this is where results are exported once you run a
     parameter estimation.

## Running an analysis (Optimization tab)

1. Click **Run Optimize**. The **Log** tab blinks — open it.
   - It reports that no reactions or coefficients were set to be optimized.
     This is the key idea: Frhodo holds every parameter constant except those
     you explicitly allow it to optimize.
2. Open the **Tables** tab (the reactions and their parameters) and expand R1,
   R2, and R3.
   - R1 and R2 show a rate constant **k**; R3 shows a fuller set of kinetic
     parameters.
     - Frhodo optimizes only the parameters the rate constant depends on. To
       optimize a rate constant directly, set its pre-exponential factor to the
       desired value, the temperature exponent to 1, and the activation energy
       to 0 — done by editing the mechanism file.
   - The displayed rate constant **k** is evaluated at the currently displayed
     shock's reactor temperature, pressure, and mixture (the same state used for
     the simulation), so it reflects any temperature or pressure dependence.
   - The bound types are **F**, **%**, **±**, **+**, and **−**: the interval
     Frhodo may optimize within. **F** is a *factor* — `F = 10` means bounds of
     value⁄10 to value×10.
     - Set R3 to **%** with a value of **10** for each of its four settings.
     - Use the mouse, or press **Tab** to cycle the fields (the **%** field is
       selected last when tabbing).
3. Optimize: return to the Optimization tab and click **Run Optimize** again.
   The simulation now moves as adjustments are tried. Watch the flashing **Log**
   tab — it should report success for both the global and local optimization.
   - Open the **Objective Function** view for fit diagnostics: a residuals QQ
     plot and a residuals density plot. A good fit has small residuals that lie
     close to the QQ-plot diagonal (approximately normally distributed).

## Results

1. The final parameters appear in the **Tables** tab — they update in real time
   during optimization, so they differ from the originals.
2. Click **Save** at the top of the window, then **Save** again in the dialog
   without changing any settings.
   - In the simulation directory you specified, a new folder appears with a
     `Sim1` subdirectory holding the optimized mechanism file (including the
     optimized rate constants).
     - Running another optimization creates `Sim2`, and so on.
     - The final mechanism after each optimization is also stored in the
       mechanism directory.
