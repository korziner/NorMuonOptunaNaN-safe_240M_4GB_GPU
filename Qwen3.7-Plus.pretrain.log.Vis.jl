#!/usr/bin/env julia
#=
    Julia-скрипт для отрисовки динамики обучения по логу pretrain.log.
    Требует пакетов: DataFrames, Plots.
    Установите через: `using Pkg; Pkg.add.(["DataFrames", "Plots"])`
=#

using DataFrames
using Plots

# ------------------------------------------------------------
# 1. Чтение и парсинг лога
# ------------------------------------------------------------
function parse_pretrain_log(path::String)
    lines = readlines(path)
    steps     = Int[]
    loss      = Float64[]
    ema       = Float64[]
    lr        = Float64[]
    gnorm     = Float64[]
    tok_s     = Int[]
    vram_free = Float64[]
    acc       = Float64[]
    entropy   = Float64[]

    # Регулярное выражение для извлечения полей из обновленного формата лога
    r = r"step\s+(\d+)/\d+\s*│\s*loss\s+([\d.e-]+)\s*│\s*ema\s+([\d.e-]+)\s*│\s*lr\s+([\d.e-]+)\s*│\s*gnorm\s+([\d.e-]+)\s*│\s*(\d+)\s+tok/s\s*│\s*VRAM\s+([\d.e-]+)\s+GB\s+free\s*│\s*acc\s+([\d.e-]+)\s*│\s*entropy\s+([\d.e-]+)"

    for line in lines
        m = match(r, line)
        if m !== nothing
            push!(steps,     parse(Int,    m[1]))
            push!(loss,      parse(Float64, m[2]))
            push!(ema,       parse(Float64, m[3]))
            push!(lr,        parse(Float64, m[4]))
            push!(gnorm,     parse(Float64, m[5]))
            push!(tok_s,     parse(Int,    m[6]))
            push!(vram_free, parse(Float64, m[7]))
            push!(acc,       parse(Float64, m[8]))
            push!(entropy,   parse(Float64, m[9]))
        end
    end

    DataFrame(; step=steps, loss=loss, ema=ema, lr=lr, gnorm=gnorm, tok_s=tok_s, vram_free=vram_free, acc=acc, entropy=entropy)
end

# ------------------------------------------------------------
# 2. Построение графиков с аннотациями
# ------------------------------------------------------------
function plot_metrics(df::DataFrame)
    # 1. Отрисовка потерь (loss) и EMA
    p1 = plot(df.step, df.loss,
              label="loss", color=:blue, lw=1.5,
              xlabel="Step", ylabel="Loss")
    plot!(p1, df.step, df.ema, label="EMA", color=:green, lw=1.5, linestyle=:dash)
    
    last_step = df.step[end]
    last_loss = df.loss[end]
    annotate!(p1, last_step, last_loss, text(" Latest loss = $last_loss", 8, :red))

    # 2. Отрисовка точности (accuracy)
    p2 = plot(df.step, df.acc,
              label="accuracy", color=:purple, lw=1.5,
              xlabel="Step", ylabel="Accuracy")
    hline!(p2, [0.5], label="random baseline", color=:gray, linestyle=:dot)

    # 3. Отрисовка градиентной нормы (gnorm)
    p3 = plot(df.step, df.gnorm,
              label="gnorm", color=:red, lw=1.5,
              xlabel="Step", ylabel="Gradient norm")
    hline!(p3, [5.0], label="danger threshold", color=:orange, linestyle=:dot)
    
    danger_idx = findall(df.gnorm .> 5.0)
    if !isempty(danger_idx)
        scatter!(p3, df.step[danger_idx], df.gnorm[danger_idx],
                 markersize=3, markercolor=:orange, label="spike >5")
    end

    # 4. Отрисовка свободной VRAM
    p4 = plot(df.step, df.vram_free,
              label="VRAM free (GB)", color=:teal, lw=1.5,
              xlabel="Step", ylabel="VRAM free (GB)")
    hline!(p4, [0.1], color=:red, linestyle=:dot, label="min safe")

    # Сборка всех графиков в одну фигуру (2×2)
    plot(p1, p2, p3, p4, layout=(2,2), size=(1280,800),
         title=["Loss & EMA" "Accuracy" "Gradient norm" "VRAM"],
         tickfontsize=8, titlefontsize=10)
end

# ------------------------------------------------------------
# 3. Исполнение
# ------------------------------------------------------------
log_path = "pretrain.log"
if isfile(log_path)
    df = parse_pretrain_log(log_path)
    if nrow(df) > 0
        plot_metrics(df)
        savefig("pretrain_dashboard.png")
        println("Графики успешно сохранены в pretrain_dashboard.png")
    else
        println("Не удалось извлечь данные из $log_path. Проверьте соответствие формата лога.")
    end
else
    println("Файл $log_path не найден.")
end
