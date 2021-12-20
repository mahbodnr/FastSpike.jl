using ..FastSpike: Network
using PlotlyJS
using GraphPlot
using Graphs

struct NetworkView
    pos_x::AbstractArray
    pos_y::AbstractArray
    edge_x::AbstractArray
    edge_y::AbstractArray
end

function vertical_layout(g)
    return zeros(nv(g)), range(-1, 1; length = nv(g))
end

function horizental_layout(g)
    return range(-1, 1; length = nv(g)), zeros(nv(g))
end

function networkView(network::Network, groups, layouts; x_distance = 1.5)
    x_loc = 0
    pos_x = []
    pos_y = []
    for (group, layout) in zip(groups, layouts)
        subgraph = SimpleGraph(group.n)
        if layout == "vertical"
            graph_x, graph_y = vertical_layout(subgraph)
        elseif layout == "horizental"
            graph_x, graph_y = horizental_layout(subgraph)
        elseif layout == "spring"
            graph_x, graph_y = spring_layout(subgraph)
        else
            error("$layout layout not supported")
        end
        graph_x .+= x_loc
        x_loc += x_distance

        pos_x = [pos_x; graph_x]
        pos_y = [pos_y; graph_y]
    end

    graph = SimpleDiGraph(network.adjacency)

    # Create plot points
    edge_x = []
    edge_y = []

    for edge in edges(graph)
        push!(edge_x, pos_x[src(edge)])
        push!(edge_x, pos_x[dst(edge)])
        push!(edge_y, pos_y[src(edge)])
        push!(edge_y, pos_y[dst(edge)])
    end

    return NetworkView(pos_x, pos_y, edge_x, edge_y)
end

function plotNetwork(view::NetworkView, spikes::AbstractVector, voltage::AbstractVector)
    # Create edges
    edges_trace = scatter(
        mode = "lines",
        x = view.edge_x,
        y = view.edge_y,
        line = attr(
            width = 0.5,
            color = "#888"
        ),
    )

    # Create nodes
    nodes_trace = scatter(
        x = view.pos_x,
        y = view.pos_y,
        mode = "markers",
        text = ["N: $i  Voltage: $(round(voltage, digits = 2))v" for (i, voltage) in enumerate(voltage)],
        marker = attr(
            showscale = true,
            colorscale = colors.imola,
            color = float(spikes),
            size = 10,
        )
    )

    # Create Plot
    plot(
        [edges_trace, nodes_trace],
        Layout(
            hovermode = "closest",
            titlefont_size = 16,
            showlegend = false,
            showarrow = false,
            xaxis = attr(showgrid = false, zeroline = false, showticklabels = false),
            yaxis = attr(showgrid = false, zeroline = false, showticklabels = false)
        )
    )

end