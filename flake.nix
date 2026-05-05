{
  description = "Markowitz mean-variance portfolio optimizer";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    uv2nix.url = "github:pyproject-nix/uv2nix";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs";
    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
    pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
    pyproject-build-systems.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    uv2nix,
    pyproject-nix,
    pyproject-build-systems,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      lib = nixpkgs.lib;
      python = pkgs.python313;
      workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};
      overlay = workspace.mkPyprojectOverlay {sourcePreference = "wheel";};
      pythonSet = (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
        ]
      );
      venv = pythonSet.mkVirtualEnv "portfolio-optimizer-env" workspace.deps.default;
    in {
      packages.default = pkgs.stdenv.mkDerivation {
        pname = "portfolio-optimizer";
        version = "0.1.0";
        src = ./.;
        dontConfigure = true;
        dontBuild = true;
        installPhase = ''
          mkdir -p $out/bin $out/lib/portfolio-optimizer
          cp -r . $out/lib/portfolio-optimizer/
          cat > $out/bin/portfolio-optimizer << EOF
          #!/bin/sh
          exec ${venv}/bin/streamlit run $out/lib/portfolio-optimizer/app.py \
            --browser.gatherUsageStats=false \
            --server.fileWatcherType=none \
            --server.port=8502 \
            "\$@"
          EOF
          chmod +x $out/bin/portfolio-optimizer
        '';

      devShells.default = pkgs.mkShell {
        packages = [
          venv
          pkgs.uv
        ];
        env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
        ];
      };
    });
}
