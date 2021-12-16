""" Write model in DynamicsTrainer to verilog-A netlist 
    input/output config format: list of tuples, (header, type, port number)
    * header: either "" or "ddt". 
    * type: either I or V 
    * port number: must be between 1 and port number 
    One tuple per model input/output. For example, ("ddt","V",2) interprets the input as the 2nd port's voltage derivative
    Similarly, ("", "I", 4) interprets the input as the 4th port's input current """
function generate_netlist_task(;
            model_name="model", 
            num_ports=0, input_config=[], output_config=[],
            time_scale=1.0, 
            input_scale, input_bias, 
            output_scale, output_bias)
    # create a function that can be directly run by a DynamicsTrainer as a callback task
    function gen_netlist(f)
        # extract parameters
        _, _ , _, p_feed, p_ode, p_link = extract_params(f)

        write_netlist(f.ode, f.link, f.feedthrough, model_name, p_ode, p_link, p_feed, num_ports, input_config, output_config;
                        time_scale=time_scale, 
                        input_scale=input_scale, 
                        input_bias=input_bias,
                        output_scale=output_scale, 
                        output_bias=output_bias,
                        dir=f.savepath);
    end
    return gen_netlist
end

""" Produces model called <model_name>_model """
function write_netlist(dyn::O, link::L, feedthrough::F, model_name::String, p_ode, p_link, p_feed, num_ports, input_config, output_config; time_scale=1.0, input_scale=ones(dyn.input_dim), input_bias=zeros(dyn.input_dim), output_scale=ones(link.output_dim), output_bias=zeros(link.output_dim), dir=".") where O<:Ode where L<:Link where F<:Feedthrough
    # dimension data
    input_dim = dyn.input_dim
    state_dim = dyn.state_dim

    # TODO: currently assumes σ_u is either identity or tanh
    if dyn.σ_u == identity
        gate_u = ""
    elseif dyn.σ_u == tanh
        gate_u = "tanh"
    else
        gate_u = ""
    end

    # odelist: list of strings, one for each dxdt
    # hidden_layers: dict of strings, one for each hidden layer unit
    # param_dict: dict of param name, param value
    odelist, hidden_layers, param_dict_ode = convert_verilogA(dyn, p_ode)

    # param_dict_output: dict of param name, param value
    # output_list: list of strings, one for each output dim
    param_dict_output, output_list = convert_verilogA(link, feedthrough, p_link, p_feed; i_scale=output_scale, i_bias=-output_scale.*output_bias)

    # write verilog A file
    # implements architecture-specific connections
    # instance specific parameter values written to the .inc netlist
    open(dir*"/"*model_name*".va", "w") do io
        # write includes
        write(io, "`include \"constants.vams\"\n")
        write(io, "`include \"disciplines.vams\"\n")
        write(io, "(* compact_module *)\n")
        # write(io, "(* ignore_hidden_state=\"all\" *)\n")

        # write inputs & outputs: ports
        iolist = ""
        for i=1:num_ports
            iolist *= "a$(i),"
        end

        # write states as inouts as well in order to set initial condition
        for i=1:state_dim
            iolist *= "x$(i),"
        end

        write(io, "module "*model_name*"("*iolist*"c);\n")

        # define inouts, with branches
        write(io, "inout "*iolist*"c;\n")    
        inlist = ""
        for i=1:input_dim
            inlist *= "_u$(i),"
            inlist *= "u$(i),"
        end
        write(io, "electrical "*iolist*inlist*"c;\n")

        for i=1:num_ports
            write(io, "branch (a$(i),c) Ia$(i);\n")
        end

        # define state variables: electrical & branch
        write(io, "electrical ")
        for i=1:state_dim-1
                write(io, "dx$(i),")
                # write(io, "x$(i),dx$(i),")
        end
        write(io, "dx$(state_dim);\n")
        # write(io, "x$(i),dx$(i);\n")

        for i=1:state_dim
            write(io, "branch (x$(i),c) Ix$(i);\n")
        end

        # write parameter definitions
        for param in keys(param_dict_ode)
            if param != "tauinv"
                write(io, "parameter real "*param*" = 0 from (-inf:inf);\n")
            else
                write(io, "parameter real "*param*" = 0 from(0:inf);\n")
            end
        end

        for param in keys(param_dict_output)
            write(io, "parameter real "*param*" = 0 from (-inf:inf);\n")
        end

        # initialize hidden layer units
	    if length(hidden_layers) > 0
            write(io, "real")
            key_list = [key for key in keys(hidden_layers)]
            for i=1:length(key_list)
                write(io, " "*key_list[i])
                if i==length(hidden_layers)
                    write(io, ";\n")
                else
                    write(io, ",")
                end
            end
        end

        # begin writing connections
        write(io, "analog begin\n")
    
        # connect inputs
        for i=1:length(input_config)
            header = input_config[i][1]
            type = input_config[i][2]
            port = input_config[i][3]

            # write(io, "V(u$(i),c)<+V(a$(i),c);\n")
            write(io, "V(_u$(i),c) <+ $(1.0/input_scale[i])*$(header)($(type)(a$(port),c)) + $(input_bias[i]);\n")
            write(io, "V(u$(i),c) <+ $(gate_u)(V(_u$(i),c));\n")
        end

        # connect derivatives
        for i=1:state_dim
            write(io, "V(dx$(i),c)<+("*string(time_scale)*")*ddt(V(x$(i),c));\n")
        end

        # define hidden layers
        for key in keys(hidden_layers)
            write(io, key * " = " * hidden_layers[key] * "\n")
        end

        # write ode
        for ode in odelist
            write(io, ode * "\n")
        end

        # write output layer
        for i=1:length(output_config)
            type = output_config[i][2]
            port = output_config[i][3]

            if type == "V"
                header = "V(a$(port),c) <+ "
            elseif type == "I"
                header = "I(Ia$(port)) <+ "
            elseif type == "O"
                write(io, "I(Ia$(port)) <+ 0.0; \n")
                continue
            else
                header = "$(type)(Ia$(port)) <+ "
            end

            write(io, header * output_list[i] * "\n")
        end

        # end
        write(io, "end // end analog\n")
        write(io, "endmodule\n")

    end # io for .va file

    #
    # write netlist file
    # instance specific parameter values written to the .inc netlist
    open(dir*"/"*model_name*".inc", "w") do io
        write(io, "// subcircuit for learned Verilog-A model\n")
        write(io, "simulator lang=spectre\n")
        write(io, "ahdl_include \"" * model_name * ".va\"\n")

        # terminal definition
        iolist = ""
        for i=1:num_ports
            iolist *= "a$(i) "
        end
        write(io, "subckt "*model_name*"_model "*iolist*"c\n")

        # # set initial condition guess to zero
        # write(io, "nodeset ")
        # for i=1:state_dim
        #     write(io, "x$(i)=0 ")
        # end
        # write(io, "\n")

        # include states in inner verilog model
        for i=1:state_dim
            iolist *= "x$(i) "
        end
        write(io, model_name*"0 ("*iolist*"c) "*model_name)

        for k in sort([string(val) for val in keys(param_dict_ode)])
            write(io, " "*k*"="*string(Float64(param_dict_ode[k])))
        end
        for k in sort([string(val) for val in keys(param_dict_output)])
            write(io, " "*k*"="*string(Float64(param_dict_output[k])))
        end
        write(io, "\n")
        write(io, "ends\n")    
    end # io for .inc file
end

""" Convert CTRNN0 to VerilogA format """
function convert_verilogA(f::CTRNN0, p)
    # unpack params
    logτ, A, B, μ, ν = unpack_params(f, p)

    # write dictionary of parameters with values
    param_dict = Dict()
    for i=1:f.state_dim
        for j=1:f.state_dim
            param_dict["A_$(i)_$(j)"] = A[i,j]
        end
        for j=1:f.input_dim
            param_dict["B_$(i)_$(j)"] = B[i,j]
        end
        param_dict["mu_$(i)"] = μ[i]
    end
    for i=1:f.state_dim
        param_dict["nu_$(i)"] = ν[i]
    end
    param_dict["tauinv"] = exp.(-logτ)[1]

    # hidden layer
    hidden_layers = Dict()
    # hidden_layer = []
    for i=1:f.state_dim
        layer_name = "h_$(i)"
        layer_string = "(mu_$(i))"
        for j=1:f.state_dim
            layer_string *= "+(A_$(i)_$(j))*(V(x$(j),c))"
        end
        for j=1:f.input_dim
            layer_string *= "+(B_$(i)_$(j))*(V(u$(j),c))"
        end

        # TODO: currently assumes σ is tanh or identity
        if f.σ == tanh
            hidden_layers[layer_name] = "tanh($(layer_string));"
        elseif f.σ == relu
            hidden_layers[layer_name] = "max(0,$(layer_string));"
        else
            hidden_layers[layer_name] = "$(layer_string);"
        end
        # push!(hidden_layer, "$(layer_name) = $(activation)($(layer_string));")
    end

    # write ode derivative assignments
    odelist = []
    for i=1:f.state_dim
        string_i = "I(Ix$(i)) <+ (V(dx$(i),c)) + (tauinv)*(V(x$(i),c)) - (nu_$(i)) - h_$(i)"

        push!(odelist, string_i*";")
    end

    return odelist, hidden_layers, param_dict
end

""" Convert CTRNN1 to VerilogA format """
function convert_verilogA(f::CTRNN1, p)
    # unpack parameters
    logτ, W, A, B, μ, ν = unpack_params(f, p)

    # write dictionary of parameters with values
    param_dict = Dict()
    for i=1:f.hidden_dim
        for j=1:f.state_dim
            param_dict["A_$(i)_$(j)"] = A[i,j]
            param_dict["W_$(j)_$(i)"] = W[j,i]
        end
        for j=1:f.input_dim
            param_dict["B_$(i)_$(j)"] = B[i,j]#/v_scale[j]
        end
        param_dict["mu_$(i)"] = μ[i]
    end
    for i=1:f.state_dim
        param_dict["nu_$(i)"] = ν[i]
    end
    param_dict["tauinv"] = exp.(-logτ)[1]

    # hidden layer
    hidden_layers = Dict()
    # hidden_layer = []
    for i=1:f.hidden_dim
        layer_name = "h_$(i)"
        layer_string = "(mu_$(i))"
        for j=1:f.state_dim
            layer_string *= "+(A_$(i)_$(j))*(V(x$(j),c))"
        end
        for j=1:f.input_dim
            layer_string *= "+(B_$(i)_$(j))*(V(u$(j),c))"
        end

        # TODO: currently assumes σ is tanh or identity
        if f.σ == tanh
            hidden_layers[layer_name] = "tanh($(layer_string));"
        elseif f.σ == relu
            hidden_layers[layer_name] = "max(0,$(layer_string));"
        else
            hidden_layers[layer_name] = "$(layer_string);"
        end
        # push!(hidden_layer, "$(layer_name) = $(activation)($(layer_string));")
    end

    # write ode derivative assignments
    odelist = []
    for i=1:f.state_dim
        string_i = "I(Ix$(i)) <+ (V(dx$(i),c)) + (tauinv)*(V(x$(i),c)) - (nu_$(i))"
        for j=1:f.hidden_dim
            string_i *= " - (W_$(i)_$(j))*h_$(j)"
        end

        push!(odelist, string_i*";")
    end

    return odelist, hidden_layers, param_dict
end

""" Convert AffineLayer to VerilogA format """
function convert_verilogA(f::AffineLayer, g::F, p_link, p_feed; v_scale=ones(f.output_dim), i_scale=ones(f.output_dim), i_bias=zeros(f.output_dim)) where F <: Feedthrough
    # TODO: assumes either tanh or identity
    if f.σ == identity
        gate_link = ""
    elseif f.σ == tanh
        gate_link = "tanh"
    else
        gate_link = ""
    end

    # unpack parameters
    C,D,b = unpack_params(f, p_link)
    D = unpack_params(g, p_feed)

    # write dictionary of parameters with values for output layer
    param_dict_output = Dict()
    for i=1:f.output_dim
        for j=1:f.state_dim
            param_dict_output["C_$(i)_$(j)"] = C[i,j]*i_scale[i]
        end
        if D != false
            for j=1:g.input_dim
                param_dict_output["D_$(i)_$(j)"] = D[i,j]*i_scale[i]
            end
        end
        param_dict_output["b_$i"] = b[i]*i_scale[i] + i_bias[i]
    end

    # write state,input -> output assignments
    output_list = []
    for i=1:f.output_dim
        string_i = "(b_$(i))"
        for j=1:f.state_dim
            string_i *= "+(C_$(i)_$(j))*$(gate_link)(V(x$(j),c))"
        end
        if D != false
            for j=1:g.input_dim
                if D[i,j] != 0.0
                    string_i *= "+(D_$(i)_$(j))*(V(u$(j),c))"
                end
            end
        end
        push!(output_list, string_i*";")
    end

    return param_dict_output, output_list
end
