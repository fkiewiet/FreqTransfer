def run_single_pair(omega_src, omega_tgt):
    print("="*80)
    print(f"### Frequency Transfer {omega_src} → {omega_tgt} ###")
    print("="*80)

    # --- 1. Load or generate dataset ---
    freq_ds_raw = get_freq_dataset(
        grid=grid,
        pml=pml_cfg,
        omega_src=omega_src,
        omega_tgt=omega_tgt,
        N_samples=N_samples,
        omega_to_k=omega_to_k,
        overwrite=False,
    )

    # --- 2. Add coordinate + omega channels ---
    freq_ds = OmegaChannelWrapper(freq_ds_raw)
    freq_ds = CoordWrapper(freq_ds, grid=grid, normalise=True)

    x0, y0 = freq_ds[0]
    in_ch = x0.shape[0]
    print(f"Input channels = {in_ch}")

    # --- 3. Define model ---
    model = SimpleFNO(
        in_ch=in_ch,
        width=48,
        modes=(12, 12),
        layers=4,
        out_ch=2,
    ).to(device)

    # --- 4. Train ---
    model, hist = train_model(
        model=model,
        dataset=freq_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=0.2,
        loss_type="mse",
        device=device,
    )

    # --- 5. Evaluate ---
    metrics = eval_relative_metrics(model, freq_ds, batch_size=batch_size, device=device)
    print("\nMetrics:", metrics)

    return {
        "omega_src": omega_src,
        "omega_tgt": omega_tgt,
        "metrics": metrics,
    }
