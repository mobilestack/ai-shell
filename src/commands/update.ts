import { command } from 'cleye';
import { execa } from 'execa';
import { dim } from 'kolorist';
import i18n from '../helpers/i18n';

export default command(
  {
    name: 'update',
    help: {
      description: 'Update AI Shell to the latest version',
    },
  },
  async () => {
    console.log('');
    const command = `npm update -g @mobilest/ai-shell`;
    console.log(dim(`${i18n.t('Running')}: ${command}`));
    console.log('');
    await execa(process.env.SHELL || '/bin/bash', ['-c', command], {
      stdio: 'inherit',
    }).catch(() => {
      // No need to handle, will go to stderr
    });
    console.log('');
  }
);
