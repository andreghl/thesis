echo "Environment Checker:"

echo "checking [Selection] env..."
python -m bargain.environments.selection

echo "checking [Proposal] env..."
python -m bargain.environments.proposal

echo "checking [Response] env..."
python -m bargain.environments.response