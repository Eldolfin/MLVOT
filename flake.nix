{
  inputs = {
    utils.url = "github:numtide/flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };
  outputs = {
    self,
    nixpkgs,
    utils,
    git-hooks,
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        precommit-config = self.checks.${system}.pre-commit-check.config;
        lib = pkgs.lib;
        python = pkgs.python313.withPackages (ppkgs:
          with ppkgs; [
            opencv-python
            numpy
            scipy

            # dev tools
            ipython
            ipdb
            # ruff
            python-lsp-server
          ]);
      in {
        devShell = let
          inherit (self.checks.${system}.pre-commit-check) shellHook enabledPackages;
        in
          pkgs.mkShell {
            shellHook =
              shellHook
              + ''
                ln -s ${python} .python
              '';
            buildInputs = enabledPackages ++ [python];
          };

        checks = {
          pre-commit-check = git-hooks.lib.${system}.run {
            src = ./.;
            package = pkgs.prek;
            hooks = {
              alejandra.enable = true;
              ruff.enable = true;
              ruff-format.enable = true;
              ty = rec {
                enable = true;
                name = "ty";
                description = "An extremely fast Python type-checker, written in Rust.";
                package = pkgs.ty;
                entry = "${lib.getExe package} check --python ${python} --extra-search-path tp1/ --extra-search-path tp2/";
                types = ["python"];
              };
              markdownlint.enable = true;
              mdsh.enable = true;
            };
          };
        };

        # Run the pre-commit hooks with `nix fmt`.
        formatter = let
          script = ''
            ${pkgs.lib.getExe precommit-config.package} run --all-files --config ${precommit-config.configFile}
          '';
        in
          pkgs.writeShellScriptBin "pre-commit-run" script;
      }
    );
}
