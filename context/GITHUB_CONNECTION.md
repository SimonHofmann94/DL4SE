mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "deine_github_mail@example.com" -f ~/.ssh/id_ed25519


eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts
chmod 644 ~/.ssh/known_hosts

cat ~/.ssh/id_ed25519.pub


ssh -T git@github.com
